$DataDir = "app\data\training_data"

if (Test-Path $DataDir) {
    if ((Get-ChildItem $DataDir | Measure-Object).Count -gt 0) {
        Write-Host "Data already exists, skipping download."
        exit 0
    }
}

Write-Host "Downloading dataset..."
kaggle datasets download -d kylescritchfield/sorted-waste-classification -p $DataDir --unzip
Write-Host "Done."