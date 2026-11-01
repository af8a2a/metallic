"""Sync, validate, and publish Documentation/Issues drafts to GitHub through the gh CLI."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ISSUES_ROOT = REPO_ROOT / 'Documentation' / 'Issues'
ISSUES_DIR = ISSUES_ROOT / 'issues'
MANIFEST_PATH = ISSUES_ROOT / 'manifest.json'
STATE_PATH = ISSUES_ROOT / 'publish-state.json'
README_PATH = ISSUES_ROOT / 'README.md'

INDEX_BEGIN = '<!-- issue-index:begin -->'
INDEX_END = '<!-- issue-index:end -->'

MARKER_FMT = '<!-- metallic-issue-id: {} -->'
MARKER_RE = re.compile(r'<!-- metallic-issue-id: (\d+) -->')
DEP_BEGIN = '<!-- dependencies:begin -->'
DEP_END = '<!-- dependencies:end -->'

LABEL_COLORS = {
    'area:shadow': '5319e7',
    'area:geometry': '0e8a16',
    'area:rhi': '1d76db',
    'area:material': 'b60205',
    'area:editor': 'fbca04',
    'area:tooling': '006b75',
    'area:build': 'c2e0c6',
    'kind:defect': 'd73a4a',
    'kind:feature': 'a2eeef',
    'kind:perf': 'f9d0c4',
    'kind:research': 'd4c5f9',
    'kind:task': 'ededed',
    'priority:P0': 'b60205',
    'priority:P1': 'd93f0b',
    'priority:P2': 'fbca04',
    'priority:P3': '0e8a16',
    'state:in-progress': '1d76db',
    'state:blocked': 'b60205',
}

MILESTONES = {
    'M1-Active': '当前正在推进：有半成品代码、默认关闭的实验候选，需要收口或决策。',
    'M2-Correctness': '正确性、确定性、验证可信度：已知缺陷与阻塞项。',
    'M3-Hardening': '生产化：默认开启、内存与生命周期、工具与测试基建补完。',
    'M4-Performance': '有测量支撑的性能跟进：已达 profile 但未采纳或未达标。',
    'M5-Research': '长期探索：尚无实现的功能与研究方向。',
}

AREAS = {'shadow', 'geometry', 'rhi', 'material', 'editor', 'tooling', 'build'}
KINDS = {'defect', 'feature', 'perf', 'research', 'task'}
PRIORITIES = {'P0', 'P1', 'P2', 'P3'}
STATES = {'in-progress', 'blocked', 'open', 'proposed'}
REQUIRED_KEYS = ('id', 'title', 'area', 'kind', 'priority', 'state', 'milestone')


@dataclass
class Draft:
    path: Path
    meta: dict[str, object]
    body: str
    number: int | None = None

    @property
    def id(self) -> str:
        return str(self.meta['id'])

    @property
    def labels(self) -> list[str]:
        return [str(item) for item in self.meta.get('labels', [])]

    @property
    def depends_on(self) -> list[str]:
        return [str(item) for item in self.meta.get('depends_on', [])]

    @property
    def milestone(self) -> str:
        return str(self.meta['milestone'])


# --- front matter -----------------------------------------------------------

def _scalar(raw: str) -> object:
    text = raw.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in '"\'':
        return text[1:-1]
    return text


def _inline_list(raw: str) -> list[str]:
    inner = raw.strip()[1:-1].strip()
    if not inner:
        return []
    return [str(_scalar(item)) for item in inner.split(',') if item.strip()]


def parse_front_matter(text: str, label: str) -> tuple[dict[str, object], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != '---':
        raise ValueError(f'{label}: missing leading "---" front matter fence')
    meta: dict[str, object] = {}
    index = 1
    while index < len(lines):
        line = lines[index]
        index += 1
        if line.strip() == '---':
            body = '\n'.join(lines[index:]).lstrip('\n')
            return meta, body
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        if line.lstrip().startswith('- '):
            key = meta.pop('__pending__', None)
            if key is None:
                raise ValueError(f'{label}: list item without owning key: {line!r}')
            meta.setdefault(key, [])
            assert isinstance(meta[key], list)
            meta[key].append(str(_scalar(line.lstrip()[2:])))
            meta['__pending__'] = key
            continue
        key, _, raw = line.partition(':')
        key = key.strip()
        value = raw.strip()
        meta.pop('__pending__', None)
        if not value:
            meta[key] = []
            meta['__pending__'] = key
        elif value.startswith('[') and value.endswith(']'):
            meta[key] = _inline_list(value)
        else:
            meta[key] = _scalar(value)
    raise ValueError(f'{label}: front matter is not closed with "---"')


def load_drafts() -> list[Draft]:
    drafts: list[Draft] = []
    for path in sorted(ISSUES_DIR.glob('*.md')):
        meta, body = parse_front_matter(path.read_text(encoding='utf-8'), path.name)
        meta.pop('__pending__', None)
        drafts.append(Draft(path=path, meta=meta, body=body))
    return drafts


# --- validation -------------------------------------------------------------

def relative_links(body: str) -> list[str]:
    """Repository-relative links a draft relies on; external URLs are not checked."""
    targets = []
    for match in re.finditer(r'\]\(([^)\s]+)\)', body):
        target = match.group(1).split('#', 1)[0]
        if not target or target.startswith(('http://', 'https://', 'mailto:')):
            continue
        targets.append(target)
    return targets


def validate(drafts: list[Draft]) -> list[str]:
    problems: list[str] = []
    seen: dict[str, Path] = {}
    if not drafts:
        problems.append(f'no issue drafts found under {ISSUES_DIR.relative_to(REPO_ROOT)}')
    for draft in drafts:
        name = draft.path.name
        missing = [key for key in REQUIRED_KEYS if not draft.meta.get(key)]
        if missing:
            problems.append(f'{name}: missing required front matter {missing}')
            continue
        if not re.fullmatch(r'\d{4}', draft.id):
            problems.append(f'{name}: id must be four digits, got {draft.id!r}')
        if draft.id in seen:
            problems.append(f'{name}: duplicate id {draft.id} (also {seen[draft.id].name})')
        seen[draft.id] = draft.path
        if not name.startswith(draft.id + '-'):
            problems.append(f'{name}: file name must start with "{draft.id}-"')
        if draft.meta['area'] not in AREAS:
            problems.append(f'{name}: unknown area {draft.meta["area"]!r}')
        if draft.meta['kind'] not in KINDS:
            problems.append(f'{name}: unknown kind {draft.meta["kind"]!r}')
        if draft.meta['priority'] not in PRIORITIES:
            problems.append(f'{name}: unknown priority {draft.meta["priority"]!r}')
        if draft.meta['state'] not in STATES:
            problems.append(f'{name}: unknown state {draft.meta["state"]!r}')
        if draft.milestone not in MILESTONES:
            problems.append(f'{name}: unknown milestone {draft.milestone!r}')
        expected = {f'area:{draft.meta["area"]}', f'kind:{draft.meta["kind"]}',
                    f'priority:{draft.meta["priority"]}'}
        if draft.meta['state'] in {'in-progress', 'blocked'}:
            expected.add(f'state:{draft.meta["state"]}')
        actual = set(draft.labels)
        if actual != expected:
            problems.append(f'{name}: labels {sorted(actual)} != expected {sorted(expected)}')
        unknown = sorted(actual - set(LABEL_COLORS))
        if unknown:
            problems.append(f'{name}: labels without a defined color {unknown}')
        evidence = draft.meta.get('evidence') or []
        if not evidence:
            problems.append(f'{name}: evidence must list at least one path')
        for item in evidence:
            entry = str(item).strip()
            if entry.startswith('git:'):
                continue  # commit/ref citation; verified by hand, not a filesystem path
            target = entry.split(':')[0].strip()
            if not (REPO_ROOT / target).exists():
                problems.append(f'{name}: evidence path does not exist: {target}')
        if '## 验收标准' not in draft.body:
            problems.append(f'{name}: body is missing a "## 验收标准" section')
        for target in relative_links(draft.body):
            if not (draft.path.parent / target).exists():
                problems.append(f'{name}: broken relative link {target}')
    for draft in drafts:
        for dependency in draft.depends_on:
            if dependency not in seen:
                problems.append(f'{draft.path.name}: depends_on unknown id {dependency!r}')
            elif dependency == draft.id:
                problems.append(f'{draft.path.name}: depends_on itself')
    return problems


def ledger_link_problems() -> list[str]:
    """README/Roadmap/Conventions links must resolve for repository readers."""
    problems = []
    for name in ('README.md', 'Roadmap.md', 'ISSUE_CONVENTIONS.md'):
        path = ISSUES_ROOT / name
        if not path.exists():
            continue
        for target in relative_links(path.read_text(encoding='utf-8')):
            if not (path.parent / target).exists():
                problems.append(f'{name}: broken relative link {target}')
    return problems


def manifest_of(drafts: list[Draft]) -> dict[str, object]:
    entries = []
    for draft in drafts:
        entries.append({
            'id': draft.id,
            'title': draft.meta['title'],
            'area': draft.meta['area'],
            'kind': draft.meta['kind'],
            'priority': draft.meta['priority'],
            'state': draft.meta['state'],
            'milestone': draft.milestone,
            'labels': draft.labels,
            'depends_on': draft.depends_on,
            'body_file': draft.path.relative_to(ISSUES_ROOT).as_posix(),
        })
    return {
        'version': 1,
        'milestones': [{'title': title, 'description': text} for title, text in MILESTONES.items()],
        'labels': [{'name': name, 'color': color} for name, color in LABEL_COLORS.items()],
        'issues': entries,
    }


# --- gh plumbing ------------------------------------------------------------

def gh(*args: str, stdin: str | None = None) -> str:
    completed = subprocess.run(['gh', *args], cwd=REPO_ROOT, input=stdin,
                               capture_output=True, text=True, encoding='utf-8', errors='replace')
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(f'gh {" ".join(args)} failed ({completed.returncode}): {detail}')
    return completed.stdout


def resolve_repo(explicit: str | None) -> str:
    """Prefer the git remote: resolving it needs no GitHub authentication."""
    if explicit:
        return explicit
    completed = subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=REPO_ROOT,
                               capture_output=True, text=True, encoding='utf-8', errors='replace')
    match = re.search(r'github\.com[:/]([^/\s]+)/([^/\s]+?)(?:\.git)?$',
                      completed.stdout.strip()) if completed.returncode == 0 else None
    if match:
        return f'{match.group(1)}/{match.group(2)}'
    data = json.loads(gh('repo', 'view', '--json', 'nameWithOwner'))
    return str(data['nameWithOwner'])


def existing_labels(repo: str) -> set[str]:
    names: set[str] = set()
    page = 1
    while True:
        data = json.loads(gh('api', f'repos/{repo}/labels?per_page=100&page={page}'))
        if not data:
            return names
        names.update(item['name'] for item in data)
        page += 1


def existing_milestones(repo: str) -> set[str]:
    titles: set[str] = set()
    page = 1
    while True:
        data = json.loads(gh('api', f'repos/{repo}/milestones?state=all&per_page=100&page={page}'))
        if not data:
            return titles
        titles.update(item['title'] for item in data)
        page += 1


def existing_issues(repo: str) -> dict[str, int]:
    """Map draft id -> issue number, recognised by the hidden body marker."""
    found: dict[str, int] = {}
    page = 1
    while True:
        raw = gh('api', f'repos/{repo}/issues?state=all&per_page=100&page={page}')
        data = json.loads(raw)
        if not data:
            return found
        for item in data:
            if 'pull_request' in item:
                continue
            match = MARKER_RE.search(item.get('body') or '')
            if match:
                found[match.group(1)] = int(item['number'])
        page += 1


def compose_body(draft: Draft, numbers: dict[str, int]) -> str:
    body = draft.body.rstrip() + '\n'
    body += f'\n---\n\n{MARKER_FMT.format(draft.id)}\n'
    resolved = [(dep, numbers[dep]) for dep in draft.depends_on if dep in numbers]
    if resolved:
        block = [DEP_BEGIN, '## 依赖 / Dependencies', '']
        block += [f'- Blocked by #{number} (`{dep}`)' for dep, number in resolved]
        block += [DEP_END]
        body = body.replace(f'\n---\n\n{MARKER_FMT.format(draft.id)}',
                            '\n' + '\n'.join(block) + f'\n\n---\n\n{MARKER_FMT.format(draft.id)}')
    return body


def create_issue(repo: str, draft: Draft, numbers: dict[str, int]) -> int:
    with tempfile.NamedTemporaryFile('w', suffix='.md', delete=False, encoding='utf-8') as handle:
        handle.write(compose_body(draft, numbers))
        body_path = Path(handle.name)
    try:
        args = ['issue', 'create', '--repo', repo, '--title', str(draft.meta['title']),
                '--body-file', str(body_path), '--milestone', draft.milestone]
        for label in draft.labels:
            args += ['--label', label]
        output = gh(*args).strip()
    finally:
        body_path.unlink(missing_ok=True)
    match = re.search(r'/(\d+)\s*$', output)
    if not match:
        raise RuntimeError(f'cannot parse issue number from gh output: {output!r}')
    return int(match.group(1))


def update_issue(repo: str, number: int, draft: Draft, numbers: dict[str, int]) -> None:
    with tempfile.NamedTemporaryFile('w', suffix='.md', delete=False, encoding='utf-8') as handle:
        handle.write(compose_body(draft, numbers))
        body_path = Path(handle.name)
    try:
        args = ['issue', 'edit', str(number), '--repo', repo, '--title', str(draft.meta['title']),
                '--body-file', str(body_path), '--milestone', draft.milestone]
        for label in draft.labels:
            args += ['--add-label', label]
        gh(*args)
    finally:
        body_path.unlink(missing_ok=True)


# --- commands ---------------------------------------------------------------

def cmd_sync(drafts: list[Draft]) -> int:
    manifest = manifest_of(drafts)
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                             encoding='utf-8')
    print(f'wrote {MANIFEST_PATH.relative_to(REPO_ROOT)} with {len(drafts)} issues')
    return 0


def cmd_check(drafts: list[Draft]) -> int:
    problems = validate(drafts)
    problems += ledger_link_problems()
    if MANIFEST_PATH.exists():
        on_disk = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
        if on_disk != manifest_of(drafts):
            problems.append('manifest.json is stale; run --sync')
    else:
        problems.append('manifest.json is missing; run --sync')
    for line in problems:
        print(f'ERROR {line}')
    if problems:
        print(f'{len(problems)} problem(s)')
        return 1
    print(f'ok: {len(drafts)} issues, manifest up to date')
    return 0


def cmd_publish(drafts: list[Draft], repo_arg: str | None, apply: bool, update: bool) -> int:
    problems = validate(drafts)
    if problems:
        for line in problems:
            print(f'ERROR {line}')
        print('refusing to publish while validation fails')
        return 1
    repo = resolve_repo(repo_arg)
    print(f'repository: {repo}')
    labels_needed = sorted({label for draft in drafts for label in draft.labels})
    milestones_needed = sorted({draft.milestone for draft in drafts})
    if not apply:
        print(f'would ensure {len(labels_needed)} labels: {", ".join(labels_needed)}')
        print(f'would ensure {len(milestones_needed)} milestones: {", ".join(milestones_needed)}')
        for draft in drafts:
            suffix = f' (blocked by {", ".join(draft.depends_on)})' if draft.depends_on else ''
            print(f'would create {draft.id} [{draft.milestone}] {draft.meta["title"]}{suffix}')
        print('dry run: rerun with --apply to create issues')
        return 0

    known_labels = existing_labels(repo)
    for name in labels_needed:
        if name in known_labels:
            continue
        gh('api', '-X', 'POST', f'repos/{repo}/labels', '-f', f'name={name}',
           '-f', f'color={LABEL_COLORS[name]}', '-f', 'description=metallic issue taxonomy')
        print(f'created label {name}')
    known_milestones = existing_milestones(repo)
    for title in milestones_needed:
        if title in known_milestones:
            continue
        gh('api', '-X', 'POST', f'repos/{repo}/milestones', '-f', f'title={title}',
           '-f', f'description={MILESTONES[title]}')
        print(f'created milestone {title}')

    numbers = existing_issues(repo)
    if numbers:
        print(f'found {len(numbers)} already published issue(s)')
    for draft in drafts:
        if draft.id in numbers:
            if not update:
                print(f'skip {draft.id} -> #{numbers[draft.id]} (already published)')
                draft.number = numbers[draft.id]
                continue
            update_issue(repo, numbers[draft.id], draft, numbers)
            print(f'updated {draft.id} -> #{numbers[draft.id]}')
        else:
            numbers[draft.id] = create_issue(repo, draft, numbers)
            print(f'created {draft.id} -> #{numbers[draft.id]}')
        draft.number = numbers[draft.id]
    # second pass: dependencies can only be rendered once every number is known
    for draft in drafts:
        if not draft.depends_on or draft.number is None:
            continue
        resolved = [numbers[dep] for dep in draft.depends_on if dep in numbers]
        if not resolved:
            continue
        update_issue(repo, draft.number, draft, numbers)
    STATE_PATH.write_text(json.dumps(numbers, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
                          encoding='utf-8')
    print(f'published {len(drafts)} issues; map written to {STATE_PATH.relative_to(REPO_ROOT)}')
    return 0


def load_published() -> dict[str, int]:
    if not STATE_PATH.exists():
        return {}
    return {str(key): int(value) for key, value in
            json.loads(STATE_PATH.read_text(encoding='utf-8')).items()}


def render_index(drafts: list[Draft]) -> str:
    published = load_published()
    lines: list[str] = []
    for title in MILESTONES:
        items = sorted((draft for draft in drafts if draft.milestone == title),
                       key=lambda draft: draft.id)
        lines.append(f'### {title} · {len(items)} 条')
        lines.append('')
        if not items:
            lines.append('_（暂无）_')
            lines.append('')
            continue
        lines.append('| ID | GitHub | 优先级 | 状态 | 标题 | 依赖 |')
        lines.append('| --- | --- | --- | --- | --- | --- |')
        for draft in items:
            deps = ', '.join(draft.depends_on) or '—'
            number = published.get(draft.id)
            gh_cell = f'#{number}' if number else '_未发布_'
            lines.append(f'| [{draft.id}](issues/{draft.path.name}) | {gh_cell} '
                         f'| {draft.meta["priority"]} | {draft.meta["state"]} '
                         f'| {draft.meta["title"]} | {deps} |')
        lines.append('')
    return '\n'.join(lines).rstrip() + '\n'


def cmd_index(drafts: list[Draft]) -> int:
    if not README_PATH.exists():
        print(f'ERROR {README_PATH.relative_to(REPO_ROOT)} is missing')
        return 1
    text = README_PATH.read_text(encoding='utf-8')
    if INDEX_BEGIN not in text or INDEX_END not in text:
        print(f'ERROR README is missing the {INDEX_BEGIN} / {INDEX_END} markers')
        return 1
    head, _, rest = text.partition(INDEX_BEGIN)
    _, _, tail = rest.partition(INDEX_END)
    README_PATH.write_text(f'{head}{INDEX_BEGIN}\n{render_index(drafts)}{INDEX_END}{tail}',
                           encoding='utf-8')
    print(f'rewrote the issue index in {README_PATH.relative_to(REPO_ROOT)} ({len(drafts)} issues)')
    return 0


def cmd_summary(drafts: list[Draft]) -> int:
    by_milestone: dict[str, list[Draft]] = {}
    for draft in drafts:
        by_milestone.setdefault(draft.milestone, []).append(draft)
    for title in MILESTONES:
        items = by_milestone.get(title, [])
        print(f'{title}: {len(items)} issue(s)')
        for draft in items:
            print(f'  {draft.id} [{draft.meta["priority"]}/{draft.meta["state"]}] {draft.meta["title"]}')
    return 0


def main() -> int:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog='modes:\n'
                                            '  --sync     rebuild manifest.json from issues/*.md\n'
                                            '  --check    validate drafts and manifest freshness\n'
                                            '  --index    rewrite the index tables in Documentation/Issues/README.md\n'
                                            '  --summary  print the milestone breakdown\n'
                                            '  (default)  dry-run publish plan; add --apply to push\n')
    parser.add_argument('--sync', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--index', action='store_true')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--apply', action='store_true', help='actually create/update GitHub issues')
    parser.add_argument('--update', action='store_true',
                        help='refresh already published issues in addition to creating new ones')
    parser.add_argument('--repo', default=None, help='owner/name; defaults to the gh-resolved repo')
    args = parser.parse_args()

    drafts = load_drafts()
    if args.sync:
        return cmd_sync(drafts)
    if args.check:
        return cmd_check(drafts)
    if args.index:
        return cmd_index(drafts)
    if args.summary:
        return cmd_summary(drafts)
    return cmd_publish(drafts, args.repo, args.apply, args.update)


if __name__ == '__main__':
    sys.exit(main())
