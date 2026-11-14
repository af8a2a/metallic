"""Source contracts for the bounded Windows CPU candidate; NOT a YAML parser.

These checks recognize this workflow/script's authored forms, not arbitrary YAML
or PowerShell semantics. They neither run the lane nor prove hosted availability,
SDK-off compatibility, first-download success, or zero native GoogleTest skips.
Run standalone: python -B tests/perf/TestCPUWorkflow.py
"""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


def code_lines(source):
    """Ignore full-line comments so comments cannot supply required commands."""
    return "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith('#'))


def contract_errors(workflow, script, requirements):
    """Check specific source shapes only; this is not general syntax validation."""
    workflow, script = code_lines(workflow), code_lines(script)
    errors = []

    def require(source, fragment, name):
        if fragment not in source:
            errors.append(name)

    require(workflow, 'on:\n  push:\n  pull_request:\n  workflow_dispatch:', 'triggers')
    require(workflow, 'permissions:\n  contents: read', 'read-only permissions')
    require(workflow, 'runs-on: windows-2025', 'runner')
    require(workflow, 'timeout-minutes: 120', 'timeout')
    require(workflow, 'uses: actions/checkout@v4', 'official checkout')
    require(workflow, 'persist-credentials: false', 'no persisted credentials')
    require(workflow, 'submodules: false', 'bounded submodules')
    require(workflow, "uses: actions/setup-python@v5\n        with:\n          python-version: '3.12'", 'Python 3.12')
    require(workflow, 'shell: pwsh\n        run: python -B tests/perf/TestCPUWorkflow.py', 'source contract step')
    require(workflow, 'shell: pwsh\n        run: ./scripts/RunCPULane.ps1', 'lane script step')
    require(workflow, 'if: always()\n        uses: actions/upload-artifact@v4', 'always upload diagnostics')
    for path in ('build-ci/ci-logs/', 'build-ci/CTestResults.xml',
                 'build-ci/Testing/Temporary/LastTest.log',
                 'build-ci/Testing/Temporary/LastTestsFailed.log'):
        require(workflow, path, f'artifact {path}')
    if 'continue-on-error:' in workflow:
        errors.append('masked workflow failure')
    if re.search(r'contents:\s*write|submodules:\s*(true|recursive)', workflow):
        errors.append('unbounded checkout or permissions')

    require(script, "$ErrorActionPreference = 'Stop'", 'PowerShell errors stop')
    require(script, '$PSNativeCommandUseErrorActionPreference = $false', 'explicit native status handling')
    require(script, 'Set-Location -LiteralPath $repositoryRoot', 'repository working directory')
    require(script, "Join-Path $repositoryRoot 'build-ci/ci-logs'", 'fixed logs directory')
    require(script, "Join-Path $repositoryRoot 'build-ci/CTestResults.xml'", 'fixed JUnit path')
    require(script, 'Microsoft Visual Studio/Installer/vswhere.exe', 'vswhere discovery')
    require(script, '-requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath', 'C++ tools discovery')
    require(script, 'Common7/Tools/VsDevCmd.bat', 'VsDevCmd path')
    require(script, 'call `"$devCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set', 'x64 environment command')
    require(script, "[Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')", 'import captured environment')
    # No intervening command may overwrite LASTEXITCODE before its checked capture.
    for command, variable, name in (
        ('& $FilePath @Arguments 2>&1 | Tee-Object -FilePath (Join-Path $logDirectory $LogName)',
         'nativeExitCode', 'logged native failure'),
        ("$installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath",
         'vswhereExitCode', 'vswhere failure'),
        ('$environment = & $env:ComSpec /d /s /c "call `"$devCmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"',
         'devCmdExitCode', 'VsDevCmd failure'),
    ):
        pattern = (re.escape(command) + r'\s+\$' + variable
                   + r' = \$LASTEXITCODE\s+if \(\$' + variable
                   + r' -ne 0\) \{\s+exit \$' + variable + r'\s+\}')
        if not re.search(pattern, script):
            errors.append(name)
    if re.search(r'Get-ChildItem\s+Env:|Write-(?:Host|Output)\s+\$environment|\$environment\s*\|', script, re.I):
        errors.append('environment disclosure')
    for fact in ('$compiler.Source', '.VersionInfo.ProductVersion', '$env:VCToolsVersion',
                 '$env:WindowsSdkDir', '$env:WindowsSDKVersion'):
        require(script, fact, f'toolchain fact {fact}')
    for tool in ('cmake', 'ninja'):
        require(script, f"Invoke-LoggedNative -FilePath {tool} -Arguments @('--version')", f'{tool} version')
    python_commands = (
        ('$pythonExecutable = (Get-Command python -ErrorAction Stop).Source', 'resolved Python'),
        ("""Invoke-LoggedNative -FilePath $pythonExecutable -Arguments @(
    '-c', 'import sys; print(sys.executable); print(sys.version); sys.exit(0 if sys.version_info[:2] == (3, 12) else "CPU lane requires Python 3.12")'
) -LogName 'python-version.log'""", 'python version'),
        ("Invoke-LoggedNative -FilePath $pythonExecutable -Arguments @('-m', 'pip', '--version') -LogName 'pip-version.log'", 'pip identity'),
        ("""Invoke-LoggedNative -FilePath $pythonExecutable -Arguments @(
    '-m', 'pip', 'install', '--only-binary=:all:', '-r', 'requirements-cpu.txt'
) -LogName 'python-install.log'""", 'CPU dependency install'),
        ("Invoke-LoggedNative -FilePath $pythonExecutable -Arguments @('-m', 'pip', 'check') -LogName 'python-check.log'", 'Python dependency consistency'),
        ("Invoke-LoggedNative -FilePath $pythonExecutable -Arguments @('-m', 'pip', 'list', '--format=freeze') -LogName 'python-packages.log'", 'resolved Python packages'),
    )
    for command, name in python_commands:
        require(script, command, name)
    configure_python = '"-DPython3_EXECUTABLE=$pythonExecutable",'
    require(script, configure_python, 'CMake Python identity')
    install = python_commands[3][0]
    configure = "Invoke-LoggedNative -FilePath cmake -Arguments @(\n    '--preset', 'metallic-ci',"
    if install in script and configure in script and script.index(install) > script.index(configure):
        errors.append('install before configure')
    if configure in script and configure_python in script:
        configure_body = script.split(configure, 1)[1].split(") -LogName 'configure.log'", 1)[0]
        require(configure_body, configure_python, 'configured Python argument')
    pins = [line.strip() for line in code_lines(requirements).splitlines() if line.strip()]
    if pins != ['numpy==2.2.6', 'matplotlib==3.10.3', 'psutil==7.0.0', 'scipy==1.15.3']:
        errors.append('CPU dependency pins')
    require(script, "Invoke-LoggedNative -FilePath cmake -Arguments @(\n    '--preset', 'metallic-ci',", 'configure preset')
    for option in ('NVIDIA_SDKS', 'SPIRV_TOOLS', 'ISPC'):
        require(script, f"'-DMETALLIC_ENABLE_{option}=OFF'", f'{option} disabled')
    require(script, ") -LogName 'configure.log'", 'configure log')
    require(script, "Invoke-LoggedNative -FilePath ./External/slang/bin/slangc.exe -Arguments @('-version')", 'Slang version')
    require(script, "Invoke-LoggedNative -FilePath cmake -Arguments @('--build', '--preset', 'metallic-ci', '--parallel', '2') -LogName 'build.log'", 'bounded preset build')
    require(script, "Invoke-LoggedNative -FilePath ctest -Arguments @(\n    '--preset', 'metallic-ci', '--output-on-failure', '--no-tests=error', '--output-junit', $junitPath\n) -LogName 'ctest.log'", 'preset CTest with JUnit')
    return errors


class CPUWorkflowSourceTests(unittest.TestCase):
    def setUp(self):
        self.workflow = (ROOT / '.github/workflows/CPU.yml').read_text(encoding='utf-8')
        self.script = (ROOT / 'scripts/RunCPULane.ps1').read_text(encoding='utf-8')
        self.requirements = (ROOT / 'requirements-cpu.txt').read_text(encoding='utf-8')

    def reject(self, source, old, new, expected):
        original = getattr(self, source)
        self.assertIn(old, original, 'mutation must actually change the source')
        mutated = original.replace(old, new, 1)
        workflow = mutated if source == 'workflow' else self.workflow
        script = mutated if source == 'script' else self.script
        requirements = mutated if source == 'requirements' else self.requirements
        self.assertIn(expected, contract_errors(workflow, script, requirements))

    def test_candidate_source_contract(self):
        self.assertEqual(contract_errors(self.workflow, self.script, self.requirements), [])

    def test_each_native_failure_is_preserved(self):
        for variable, error in (('nativeExitCode', 'logged native failure'),
                                ('vswhereExitCode', 'vswhere failure'),
                                ('devCmdExitCode', 'VsDevCmd failure')):
            with self.subTest(variable=variable):
                self.reject('script', f'exit ${variable}', 'exit 0', error)
                self.reject('script', f'exit ${variable}', '', error)
                self.reject('script', f'${variable} = $LASTEXITCODE', '', error)
                self.reject('script', f'if (${variable} -ne 0)', f'if (${variable} -lt 0)', error)
                self.reject('script', f'${variable} = $LASTEXITCODE',
                            f'Write-Host ignored\n${variable} = $LASTEXITCODE', error)

    def test_missing_total_sdk_shutdown(self):
        for option in ('NVIDIA_SDKS', 'SPIRV_TOOLS', 'ISPC'):
            with self.subTest(option=option):
                self.reject('script', f"'-DMETALLIC_ENABLE_{option}=OFF'",
                            f"'-DMETALLIC_ENABLE_{option}=ON'", f'{option} disabled')
                self.reject('script', f"'-DMETALLIC_ENABLE_{option}=OFF'", '', f'{option} disabled')

    def test_configure_preset_cannot_drift(self):
        self.reject('script', "'--preset', 'metallic-ci',\n", "'--preset', 'metallic-dev',\n", 'configure preset')

    def test_build_preset_and_parallelism_cannot_drift(self):
        self.reject('script', "'--build', '--preset', 'metallic-ci'",
                    "'--build', '--preset', 'metallic-full'", 'bounded preset build')
        self.reject('script', "'--parallel', '2'", "'--parallel', '0'", 'bounded preset build')

    def test_test_preset_cannot_drift(self):
        self.reject('script', "'--preset', 'metallic-ci', '--output-on-failure'",
                    "'--preset', 'other', '--output-on-failure'", 'preset CTest with JUnit')

    def test_empty_selection_and_junit_cannot_disappear(self):
        self.reject('script', "'--no-tests=error', ", '', 'preset CTest with JUnit')
        self.reject('script', "'--output-junit', $junitPath", '', 'preset CTest with JUnit')
        self.reject('script', 'build-ci/CTestResults.xml', 'results.xml', 'fixed JUnit path')

    def test_artifacts_survive_failure(self):
        self.reject('workflow', 'if: always()', 'if: success()', 'always upload diagnostics')
        for path in ('build-ci/ci-logs/', 'build-ci/CTestResults.xml',
                     'build-ci/Testing/Temporary/LastTest.log',
                     'build-ci/Testing/Temporary/LastTestsFailed.log'):
            with self.subTest(path=path):
                self.reject('workflow', path, '', f'artifact {path}')

    def test_missing_logs_rejected(self):
        self.reject('script', "-LogName 'configure.log'", "-LogName 'other.log'", 'configure log')
        self.reject('script', "-LogName 'build.log'", "-LogName 'other.log'", 'bounded preset build')
        self.reject('script', "-LogName 'ctest.log'", "-LogName 'other.log'", 'preset CTest with JUnit')

    def test_vsdevcmd_failure_not_hidden_by_set(self):
        self.reject('script', '>nul && set', '>nul & set', 'x64 environment command')
        self.reject('script', '-arch=x64 -host_arch=x64', '-arch=x86 -host_arch=x86', 'x64 environment command')

    def test_environment_not_printed(self):
        script = self.script + '\n$environment | Write-Output\n'
        self.assertIn('environment disclosure', contract_errors(self.workflow, script, self.requirements))

    def test_optional_assets_and_credentials_not_enabled(self):
        self.reject('workflow', 'submodules: false', 'submodules: recursive', 'bounded submodules')
        self.reject('workflow', 'persist-credentials: false', 'persist-credentials: true', 'no persisted credentials')
        self.reject('workflow', 'contents: read', 'contents: write', 'read-only permissions')

    def test_workflow_failures_not_masked(self):
        workflow = self.workflow + '\n    continue-on-error: true\n'
        self.assertIn('masked workflow failure', contract_errors(workflow, self.script, self.requirements))

    def test_source_comments_cannot_supply_sdk_switch(self):
        self.reject('script', "    '-DMETALLIC_ENABLE_NVIDIA_SDKS=OFF',",
                    "    # '-DMETALLIC_ENABLE_NVIDIA_SDKS=OFF',", 'NVIDIA_SDKS disabled')

    def test_tool_versions_are_reported(self):
        self.reject('script', '.VersionInfo.ProductVersion', '.Name', 'toolchain fact .VersionInfo.ProductVersion')
        for tool in ('cmake', 'ninja'):
            with self.subTest(tool=tool):
                self.reject('script', f"-FilePath {tool} -Arguments @('--version')", '', f'{tool} version')
        self.reject('script', "-LogName 'python-version.log'", '', 'python version')

    def test_standalone_check_is_wired(self):
        self.reject('workflow', 'run: python -B tests/perf/TestCPUWorkflow.py', 'run: echo omitted', 'source contract step')
        self.reject('workflow', 'run: ./scripts/RunCPULane.ps1', 'run: echo omitted', 'lane script step')

    def test_cpu_manifest_pins_are_bounded(self):
        for pin in ('numpy==2.2.6', 'matplotlib==3.10.3', 'psutil==7.0.0', 'scipy==1.15.3'):
            with self.subTest(pin=pin):
                self.reject('requirements', pin, '', 'CPU dependency pins')
                self.reject('requirements', pin, pin.replace('==', '>='), 'CPU dependency pins')
        self.assertIn('CPU dependency pins', contract_errors(
            self.workflow, self.script, self.requirements + '\nOpenEXR>=3.2\n'))

    def test_dependency_install_cannot_disappear_or_use_other_python(self):
        for old, new in (("'-m', 'pip', 'install'", "'-m', 'pip', '--version'"),
                         ("'requirements-cpu.txt'", "'Tools/requirements-terrain.txt'"),
                         ("'--only-binary=:all:', ", ''),
                         ("-LogName 'python-install.log'", '')):
            with self.subTest(old=old):
                self.reject('script', old, new, 'CPU dependency install')
        self.reject('script', "Invoke-LoggedNative -FilePath $pythonExecutable -Arguments @(\n    '-m', 'pip', 'install'",
                    "Invoke-LoggedNative -FilePath python -Arguments @(\n    '-m', 'pip', 'install'", 'CPU dependency install')

    def test_interpreter_resolution_and_version_cannot_drift(self):
        self.reject('script', '(Get-Command python -ErrorAction Stop).Source', "'python-other.exe'", 'resolved Python')
        self.reject('script', 'sys.version_info[:2] == (3, 12)', 'sys.version_info[:2] == (3, 11)', 'python version')
        self.reject('script', 'print(sys.executable); ', '', 'python version')

    def test_cmake_must_use_the_installed_interpreter(self):
        self.reject('script', '"-DPython3_EXECUTABLE=$pythonExecutable",', '', 'CMake Python identity')
        self.reject('script', '"-DPython3_EXECUTABLE=$pythonExecutable",', "'-DPython3_EXECUTABLE=python-other.exe',", 'CMake Python identity')

    def test_install_precedes_configure(self):
        start = self.script.index('Invoke-LoggedNative -FilePath $pythonExecutable -Arguments @(\n    \'-m\', \'pip\', \'install\'')
        end = self.script.index(") -LogName 'python-install.log'", start) + len(") -LogName 'python-install.log'")
        install = self.script[start:end]
        script = self.script[:start] + self.script[end:] + '\n' + install
        self.assertIn('install before configure', contract_errors(self.workflow, script, self.requirements))

    def test_dependency_verification_and_resolution_are_logged(self):
        for fragment, error in (("'-m', 'pip', '--version'", 'pip identity'),
                                ("'-m', 'pip', 'check'", 'Python dependency consistency'),
                                ("'-m', 'pip', 'list', '--format=freeze'", 'resolved Python packages')):
            with self.subTest(fragment=fragment):
                self.reject('script', fragment, "'-m', 'pip', '--help'", error)

    def test_comments_cannot_supply_install_or_pins(self):
        self.reject('script', "    '-m', 'pip', 'install',", "    # '-m', 'pip', 'install',", 'CPU dependency install')
        self.reject('requirements', 'scipy==1.15.3', '# scipy==1.15.3', 'CPU dependency pins')


if __name__ == '__main__':
    unittest.main()
