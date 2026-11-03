#!/usr/bin/env bash
# Fix the G6K relay pin mapping in Isaac's shared Analog_Channel schematic.
# Requires Bash, Python 3.8+, and KiCad 10 to reopen the supplied project.
# Preview: bash fix_relay_footprints.sh /path/to/project/Analog_Channel.kicad_sch
# Apply:   bash fix_relay_footprints.sh /path/to/project/Analog_Channel.kicad_sch --apply
# Keep the shared sheet beside the project's .kicad_pro file for this script.
# Close KiCad first. This DOES NOT edit PCB copper or fix other board issues.
# Physical G6K-2F-Y pads: coil 1/8; COM/NC/NO = 3/2/4 and 6/7/5.
set -euo pipefail
command -v python3 >/dev/null 2>&1 || { echo 'Python 3 is required.' >&2; exit 1; }
exec python3 - "$@" <<'PY'
import argparse
import collections
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

LIB = 'Hydrophone_RelayFix'
FP = 'Relay_SMD:Relay_DPDT_Omron_G6K-2F-Y'
SPECS = {
    'Relay:Relay_DPDT': (
        'G6K_2F_Y_DPDT',
        {'A2':'1', '14':'2', '12':'3', '11':'4', '21':'5', '22':'6', '24':'7', 'A1':'8'},
        {'A2':'1', '14':'4', '12':'2', '11':'3', '21':'6', '22':'7', '24':'5', 'A1':'8'}),
    'Relay:Relay_SPST-NO': (
        'G6K_2F_Y_SPST_NO',
        {'A2':'1', '14':'2', '13':'4', 'A1':'8'},
        {'A2':'1', '14':'4', '13':'3', 'A1':'8'}),
}

def require(ok, message):
    if not ok:
        raise ValueError(message)

class Atom:
    def __init__(self, value, start, end):
        self.value, self.start, self.end = value, start, end

class Node:
    def __init__(self, items, start, end):
        self.items, self.start, self.end = items, start, end

def atom(node, index=1):
    value = node.items[index]
    require(isinstance(value, Atom), 'Unexpected nested value.')
    return value

def tag(node):
    return atom(node, 0).value

def children(node, key):
    return [x for x in node.items if isinstance(x, Node) and tag(x) == key]

def child(node, key):
    found = children(node, key)
    require(len(found) == 1, 'Expected exactly one ' + key)
    return found[0]

def value(node, key):
    return atom(child(node, key)).value

def prop(node, key):
    found = [x for x in children(node, 'property') if atom(x).value == key]
    require(len(found) == 1, 'Missing or duplicate property: ' + key)
    return atom(found[0], 2)

def parse(text):
    # Track original offsets so unrelated text/formatting is left byte-for-byte intact.
    stack, roots, pos = [], [], 0
    for match in re.finditer(r'"(?:\\.|[^"\\])*"|[()]|[^\s()"]+', text):
        require(not text[pos:match.start()].strip(), 'Invalid S-expression token.')
        pos = match.end()
        token = match.group()
        if token == '(':
            stack.append(([], match.start()))
        elif token == ')':
            require(bool(stack), 'Unexpected closing parenthesis.')
            items, start = stack.pop()
            require(bool(items), 'Empty S-expression.')
            node = Node(items, start, match.end())
            (stack[-1][0] if stack else roots).append(node)
        else:
            require(bool(stack), 'Atom outside S-expression.')
            decoded = json.loads(token) if token.startswith('"') else token
            stack[-1][0].append(Atom(decoded, match.start(), match.end()))
    require(not stack and len(roots) == 1 and not text[pos:].strip(), 'Incomplete S-expression.')
    return roots[0]

def edit_atom(edits, item, new_value):
    if item.value != new_value:
        edits.append((item.start, item.end, json.dumps(new_value)))

def patched(text, edits):
    last = len(text)
    for start, end, replacement in sorted(edits, reverse=True):
        require(end <= last, 'Overlapping edits.')
        text = text[:start] + replacement + text[end:]
        last = start
    return text

def pins(symbol):
    return [p for unit in children(symbol, 'symbol') for p in children(unit, 'pin')]

def pin_map(symbol):
    ps = pins(symbol)
    result = {value(p, 'name'): value(p, 'number') for p in ps}
    require(len(result) == len(ps), 'Duplicate relay pin names.')
    require(len(set(result.values())) == len(ps), 'Duplicate relay pin numbers.')
    return result

def walk(node, key):
    for item in node.items:
        if isinstance(item, Node):
            if tag(item) == key:
                yield item
            yield from walk(item, key)

def atomic_write(path, data):
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, (path.stat().st_mode & 0o777) if path.exists() else 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

def main():
    ap = argparse.ArgumentParser(description='Preview or apply the reviewed G6K relay mapping fix. PCB routing is a separate step.')
    ap.add_argument('schematic', type=Path, help='Shared Analog_Channel.kicad_sch, in the project folder')
    ap.add_argument('--apply', action='store_true', help='Write changes after creating a timestamped backup')
    args = ap.parse_args()
    path = args.schematic.expanduser().resolve(strict=True)
    require(path.suffix == '.kicad_sch', 'Pass a .kicad_sch file, not the PCB.')
    folder = path.parent
    require(any(folder.glob('*.kicad_pro')), 'The shared sheet must be beside its .kicad_pro project file.')
    if args.apply:
        locks = list(folder.glob('*.lck'))
        require(not locks, 'Close KiCad before applying. Lock files found: ' + ', '.join(p.name for p in locks))
    original = path.read_bytes()
    text = original.decode('utf-8')
    root = parse(text)
    require(tag(root) == 'kicad_sch', 'Not a KiCad schematic.')
    cached = children(child(root, 'lib_symbols'), 'symbol')
    instances = children(root, 'symbol')
    edits, library_symbols, changed_instances, reports = [], [], [], []
    for old_id, (short, old_map, good_map) in SPECS.items():
        new_id = LIB + ':' + short
        targets = [s for s in instances if value(s, 'lib_id') in (old_id, new_id)]
        require(bool(targets), 'Expected relay type absent: ' + old_id + '. Use the reviewed shared analog sheet.')
        ids = {value(s, 'lib_id') for s in targets}
        require(len(ids) == 1, 'Mixed old/new relay types; inspect manually.')
        current_id = next(iter(ids))
        defs = [s for s in cached if atom(s).value == current_id]
        require(len(defs) == 1, 'Missing/duplicate embedded relay definition.')
        require(not any(atom(s).value == new_id for s in cached if s is not defs[0]), 'Conflicting corrected definition.')
        definition = defs[0]
        actual_map = pin_map(definition)
        allowed = (good_map,) if current_id == new_id else (old_map, good_map)
        require(actual_map in allowed, 'Unexpected relay mapping for ' + current_id + '; refusing to guess.')
        number_map = {actual_map[name]: good_map[name] for name in good_map}
        for symbol in targets:
            reference = prop(symbol, 'Reference').value
            require(prop(symbol, 'Footprint').value == FP, reference + ': unexpected footprint.')
            ipins = children(symbol, 'pin')
            require(collections.Counter(atom(p).value for p in ipins) == collections.Counter(actual_map.values()), reference + ': unexpected instance pin list.')
            for pin in ipins:
                edit_atom(edits, atom(pin), number_map[atom(pin).value])
            edit_atom(edits, atom(child(symbol, 'lib_id')), new_id)
            edit_atom(edits, prop(symbol, 'Value'), 'G6K-2F-Y')
            refs = sorted({atom(n).value for n in walk(symbol, 'reference')}) or [reference]
            reports.append(', '.join(refs))
            changed_instances.append(symbol)
        # Build a corrected cached definition and a matching reusable project symbol.
        local_text = text[definition.start:definition.end]
        local = parse(local_text)
        local_edits = []
        edit_atom(local_edits, atom(local), new_id)
        previous_short = current_id.split(':', 1)[1]
        for unit in children(local, 'symbol'):
            unit_name = atom(unit).value
            require(unit_name.startswith(previous_short + '_'), 'Unexpected relay unit name.')
            edit_atom(local_edits, atom(unit), short + unit_name[len(previous_short):])
        for pin in pins(local):
            edit_atom(local_edits, atom(child(pin, 'number')), good_map[value(pin, 'name')])
        edit_atom(local_edits, prop(local, 'Footprint'), FP)
        edit_atom(local_edits, prop(local, 'Value'), 'G6K-2F-Y')
        edit_atom(local_edits, prop(local, 'ki_fp_filters'), 'Relay_DPDT_Omron_G6K-2F-Y')
        fixed = patched(local_text, local_edits)
        if fixed != local_text:
            edits.append((definition.start, definition.end, fixed))
        library_node = parse(fixed)
        library_edits = []
        edit_atom(library_edits, atom(library_node), short)
        library_symbols.append(patched(fixed, library_edits))

    output = patched(text, edits)
    checked = parse(output)
    fixed_defs = {atom(s).value:s for s in children(child(checked, 'lib_symbols'), 'symbol')}
    for _, (short, _, good_map) in SPECS.items():
        require(pin_map(fixed_defs[LIB + ':' + short]) == good_map, 'Post-edit pin verification failed.')
    lib_path, table_path = folder / (LIB + '.kicad_sym'), folder / 'sym-lib-table'
    # 20231120 is a supported symbol-library container version. KiCad 10 is
    # required for the source project's newer symbol properties and schematic.
    lib_text = '(kicad_symbol_lib (version 20231120) (generator "relay_fix")\n' + '\n'.join(library_symbols) + '\n)\n'
    parse(lib_text)
    require(not lib_path.exists() or lib_path.read_bytes() == lib_text.encode(), 'Existing ' + lib_path.name + ' differs; refusing to overwrite it.')
    table_text = table_path.read_text(encoding='utf-8') if table_path.exists() else '(sym_lib_table\n  (version 7)\n)\n'
    table = parse(table_text)
    require(tag(table) == 'sym_lib_table', 'Invalid sym-lib-table.')
    entries = [n for n in children(table, 'lib') if value(n, 'name') == LIB]
    uri = '${KIPRJMOD}/' + LIB + '.kicad_sym'
    require(len(entries) <= 1, 'Duplicate library table entry.')
    if entries:
        require(value(entries[0], 'uri') == uri and value(entries[0], 'type') == 'KiCad', 'Conflicting library table entry.')
    else:
        entry = '  (lib (name "' + LIB + '")(type "KiCad")(uri "' + uri + '")(options "")(descr "Reviewed G6K relay pin mapping"))\n'
        table_text = table_text[:table.end - 1] + entry + table_text[table.end - 1:]
    parse(table_text)
    proposed = {path:output.encode('utf-8'), lib_path:lib_text.encode('utf-8'), table_path:table_text.encode('utf-8')}
    writes = {p:data for p,data in proposed.items() if not p.exists() or p.read_bytes() != data}
    print('Verified relay references: ' + '; '.join(reports))
    print('Correct pads: coil 1/8; pole 1 COM=3 NC=2 NO=4; pole 2 COM=6 NC=7 NO=5.')
    print('SPST drawings use COM=3 and NO=4; remaining contact pads are unused.')
    if not writes:
        print('Already fixed. No files changed.')
        return
    print('Files to write: ' + ', '.join(p.name for p in writes))
    if not args.apply:
        print('DRY RUN: no files changed. Close KiCad and rerun with --apply to write.')
        return
    # Save every original before writing any destination. An interrupted process
    # can be recovered using the backup; normal write errors roll back automatically.
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = Path(tempfile.mkdtemp(prefix='relay-fix-backup-' + stamp + '-', dir=str(folder)))
    originals = {p:p.read_bytes() if p.exists() else None for p in writes}
    for p,data in originals.items():
        if data is not None:
            shutil.copy2(p, backup / p.name)
    restore = ['Close KiCad before restoring.', 'Restore these files from this folder:']
    restore += [p.name for p,data in originals.items() if data is not None]
    restore += ['These files were newly created; remove them when restoring:']
    restore += [p.name for p,data in originals.items() if data is None]
    (backup / 'RESTORE.txt').write_text('\n'.join(restore) + '\n', encoding='utf-8')
    try:
        for p,data in writes.items():
            atomic_write(p, data)
    except BaseException:
        for p,data in originals.items():
            if data is None:
                if p.exists():
                    p.unlink()
            else:
                atomic_write(p, data)
        raise
    print('Applied. Backup: ' + str(backup))
    print('Next: reopen the top-level project; Update PCB from Schematic (F8).')
    print('Inspect and reroute affected relay connections, refill zones, then run ERC/DRC.')
    print('PCB copper was not edited. This fixes only the reviewed relay pin mapping.')

try:
    main()
except (ValueError, OSError, IndexError, KeyError) as error:
    print('ERROR: ' + str(error), file=sys.stderr)
    sys.exit(1)
PY
