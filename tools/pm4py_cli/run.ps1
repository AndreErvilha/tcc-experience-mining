# run.ps1
# Roda o pm4py-cli usando o python.exe REAL, contornando o shim .bat do
# pyenv-win (python.bat / pm4py-cli.bat), que corrompe argumentos contendo
# caracteres especiais como [ ] ( ) : quando passam pelo cmd.exe.
#
# Uso (a partir da raiz do projeto):
#   .\run.ps1 analysis filter --csv "..." --trace_include "..." "[A]" "..." "[B]" "..."
#
# Ou de qualquer outra pasta:
#   & "C:\caminho\para\jornada_ux_cli\run.ps1" analysis filter ...

$ErrorActionPreference = 'Stop'

$python = $null

# 1) Tenta obter o python real pelo próprio pyenv (sem caracteres especiais, ok)
try {
    $python = (& pyenv which python 2>$null | Select-Object -First 1)
} catch {
    $python = $null
}

# 2) Fallback: procura o python.exe dentro de ~/.pyenv/pyenv-win/versions
if (-not $python -or -not (Test-Path $python)) {
    $versionsRoot = Join-Path $env:USERPROFILE '.pyenv\pyenv-win\versions'
    if (Test-Path $versionsRoot) {
        $python = Get-ChildItem -Path $versionsRoot -Recurse -Filter 'python.exe' -ErrorAction SilentlyContinue |
                  Sort-Object LastWriteTime -Descending |
                  Select-Object -First 1 -ExpandProperty FullName
    }
}

# 3) Último recurso: qualquer python.exe disponível no PATH
if (-not $python) {
    $python = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
}

if (-not $python -or -not (Test-Path $python)) {
    Write-Host 'ERRO: não foi possível localizar um python.exe (execute "pyenv which python").' -ForegroundColor Red
    exit 1
}

& $python (Join-Path $PSScriptRoot 'run.py') @args
exit $LASTEXITCODE
