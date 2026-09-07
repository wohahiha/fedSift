param(
    [ValidateSet('verify','train-smoke','rebuild','replay','benchmark','search','audit-search','tests')]
    [string]$Mode = 'verify',
    [string]$Distribution = 'Ubuntu',
    [switch]$ModelsOnly
)
$ErrorActionPreference = 'Stop'
$linuxRoot = & wsl.exe -d $Distribution --exec wslpath -u $PSScriptRoot.Replace('\','/')
if ($LASTEXITCODE -ne 0 -or -not $linuxRoot) { throw '无法转换实验目录的 WSL 路径' }
$arguments = @(($linuxRoot.Trim() + '/run.sh'), $Mode)
if ($ModelsOnly) {
    if ($Mode -ne 'verify') { throw 'ModelsOnly 仅适用于 verify' }
    $arguments += '--models-only'
}
& wsl.exe -d $Distribution --exec bash @arguments
exit $LASTEXITCODE
