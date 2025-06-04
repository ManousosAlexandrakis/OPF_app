// Global variables
let uploadedFiles = [];
let currentChart = null;

// DOM elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const plotBtn = document.getElementById('plotBtn');
const clearBtn = document.getElementById('clearBtn');
const fileList = document.getElementById('fileList');
const chartCanvas = document.getElementById('resultsChart');

// Event listeners
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('highlight');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('highlight');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('highlight');
    handleFiles(e.dataTransfer.files);
});

fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
    fileInput.value = ''; // Reset input
});

plotBtn.addEventListener('click', () => {
    const plotType = document.getElementById('plotType').value;
    generatePlot(plotType);
});

clearBtn.addEventListener('click', clearAll);

// File handling
function handleFiles(files) {
    Array.from(files).forEach(file => {
        if (file.name.endsWith('.xlsx')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const data = new Uint8Array(e.target.result);
                const workbook = XLSX.read(data, { type: 'array' });

                // Determine model type from filename
                let modelType;
                const lowerName = file.name.toLowerCase();

                if (lowerName.includes('btheta')) {
                    modelType = 'btheta';
                } else if (lowerName.includes('bolognani')) {
                    modelType = 'bolognani';
                } else if (lowerName.includes('decoupled')) {
                    modelType = 'decoupled';
                } else if (lowerName.includes('ac')) {
                    modelType = 'ac';
                } else {
                    alert(`Cannot determine model type from filename: ${file.name}\nPlease include 'btheta', 'bolognani', 'decoupled', or 'ac' in the filename.`);
                    return;
                }

                uploadedFiles.push({
                    name: file.name,
                    model: modelType,
                    data: workbook
                });
                updateFileList();
            };
            reader.readAsArrayBuffer(file);
        }
    });
}

function updateFileList() {
    fileList.innerHTML = '';
    uploadedFiles.forEach((file, index) => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <span>${file.name} (${file.model})</span>
            <button onclick="removeFile(${index})">×</button>
        `;
        fileList.appendChild(fileItem);
    });
}

function removeFile(index) {
    uploadedFiles.splice(index, 1);
    updateFileList();
}

// Data extraction functions
function extractACData(workbook, plotType) {
    try {
        let sheetName, valueColumn, busColumn;
        
        switch(plotType) {
            case 'active_power':
                sheetName = 'prod';
                valueColumn = 'p';
                busColumn = 'bus';
                break;
            case 'reactive_power':
                sheetName = 'reactive';
                valueColumn = 'q_pu';
                busColumn = 'bus';
                break;
            case 'voltage':
                sheetName = 'bus';
                valueColumn = 'vm_pu';
                busColumn = 'Bus';
                break;
            case 'voltage_angle':
                sheetName = 'bus';
                valueColumn = 'va_degree';
                busColumn = 'Bus';
                break;
            case 'price':
                sheetName = 'LMP';
                valueColumn = 'node_price';
                busColumn = 'bus';
                break;
            default:
                return null;
        }

        const sheet = workbook.Sheets[sheetName];
        if (!sheet) {
            console.warn(`AC Model: No '${sheetName}' sheet found`);
            return null;
        }

        const data = XLSX.utils.sheet_to_json(sheet);
        const actualBusColumn = data.some(row => busColumn in row) ? busColumn : 
                               busColumn === 'Bus' ? 'bus' : 'Bus';
        
        return data.map(row => ({
            bus: row[actualBusColumn].toString(),
            value: row[valueColumn] !== undefined ? parseFloat(row[valueColumn]) : null
        })).filter(item => item.value !== null);
        
    } catch (e) {
        console.error(`Error extracting AC model data for ${plotType}:`, e);
        return null;
    }
}

function extractBolognaniData(workbook, plotType) {
    try {
        let sheetName, valueColumn;
        
        switch(plotType) {
            case 'active_power':
                sheetName = 'Production';
                valueColumn = 'production';
                break;
            case 'reactive_power':
                sheetName = 'Reactive_Production';
                valueColumn = 'q_pu';
                break;
            case 'voltage':
                sheetName = 'Results';
                valueColumn = 'vm_pu';
                break;
            case 'voltage_angle':
                sheetName = 'Results';
                valueColumn = 'va_degree';
                break;
            case 'price':
                sheetName = 'Price';
                valueColumn = 'price';
                break;
            default:
                return null;
        }

        const sheet = workbook.Sheets[sheetName];
        if (!sheet) return null;

        const data = XLSX.utils.sheet_to_json(sheet);
        const busColumn = data.some(row => 'bus' in row) ? 'bus' : 'Bus';
        
        return data.map(row => ({
            bus: row[busColumn].toString(),
            value: row[valueColumn] !== undefined ? parseFloat(row[valueColumn]) : null
        })).filter(item => item.value !== null);

    } catch (e) {
        console.error(`Error extracting Bolognani model data for ${plotType}:`, e);
        return null;
    }
}

function extractTraditionalData(workbook, plotType, modelType) {
    try {
        let sheetName, valueColumn;
        
        switch(plotType) {
            case 'active_power':
                sheetName = 'Production';
                valueColumn = 'production';
                break;
            case 'reactive_power':
                sheetName = 'Production';
                valueColumn = 'q';
                break;
            case 'voltage':
                sheetName = 'Results';
                valueColumn = 'V_pu';
                break;
            case 'voltage_angle':
                sheetName = 'Results';
                valueColumn = 'Delta';
                break;
            case 'price':
                sheetName = 'Price';
                valueColumn = 'node_price';
                break;
            default:
                return null;
        }

        const sheet = workbook.Sheets[sheetName];
        if (!sheet) return null;

        const data = XLSX.utils.sheet_to_json(sheet);
        const busColumn = data.some(row => 'Bus' in row) ? 'Bus' : 'bus';
        
        return data.map(row => ({
            bus: row[busColumn].toString(),
            value: row[valueColumn] !== undefined ? parseFloat(row[valueColumn]) : null
        })).filter(item => item.value !== null);

    } catch (e) {
        console.error(`Error extracting ${modelType} model data for ${plotType}:`, e);
        return null;
    }
}

function extractPlotData(workbook, modelType, plotType) {
    switch(modelType) {
        case 'ac':
            return extractACData(workbook, plotType);
        case 'bolognani':
            return extractBolognaniData(workbook, plotType);
        case 'btheta':
        case 'decoupled':
            return extractTraditionalData(workbook, plotType, modelType);
        default:
            return null;
    }
}

// Plot generation
function generatePlot(plotType = 'active_power') {
    if (uploadedFiles.length === 0) {
        alert('Please upload at least one result file');
        return;
    }

    const busNumbers = new Set();
    const datasets = [];

    // First pass: collect unique bus numbers
    uploadedFiles.forEach(file => {
        const plotData = extractPlotData(file.data, file.model, plotType);
        if (!plotData) return;

        plotData.forEach(item => {
            busNumbers.add(item.bus);
        });
    });

    const sortedBusNumbers = Array.from(busNumbers).sort((a, b) => parseInt(a) - parseInt(b));

    if (sortedBusNumbers.length === 0) {
        alert(`No ${plotType} data found in uploaded files`);
        return;
    }

    // Second pass: create datasets
    uploadedFiles.forEach(file => {
        const plotData = extractPlotData(file.data, file.model, plotType);
        if (!plotData) return;

        const dataMap = {};
        plotData.forEach(item => {
            dataMap[item.bus] = item.value;
        });

        const datasetConfig = {
    label: file.model === "ac" ? "AC" : file.model.charAt(0).toUpperCase() + file.model.slice(1),
    backgroundColor: getModelColor(file.model),
    borderColor: getModelColor(file.model),
    borderWidth: 2,
    data: sortedBusNumbers.map(bus => dataMap[bus] ?? null)
};

        if (plotType === 'voltage' || plotType === 'voltage_angle' || plotType === 'price') {
            Object.assign(datasetConfig, {
                pointBackgroundColor: getModelColor(file.model),
                pointRadius: file.model === 'bolognani' || file.model === 'ac' ? 5 : 4,
                pointHoverRadius: file.model === 'bolognani' ? 7 : 6,
                pointStyle: file.model === 'btheta' ? 'rect' : 'circle',
                borderDash: file.model === 'decoupled' ? [5, 5] : [],
                showLine: true,
                lineTension: 0,
                fill: false
            });
        } else {
            Object.assign(datasetConfig, {
                borderWidth: 1,
                borderRadius: file.model === 'bolognani' ? 3 : 0
            });
        }

        datasets.push(datasetConfig);
    });

    if (currentChart) {
        currentChart.destroy();
    }

    const chartType = (plotType === 'voltage' || plotType === 'voltage_angle' || plotType === 'price') 
        ? 'scatter' 
        : 'bar';

    const yAxisTitle = {
        'active_power': 'Active Power (MW)',
        'reactive_power': 'Reactive Power (MVAr)',
        'voltage': 'Voltage Magnitude (pu)',
        'voltage_angle': 'Voltage Angle (degrees)',
        'price': 'Price ($/MWh)'
    }[plotType];

    const chartTitle = {
        'active_power': 'Active Power Production Comparison',
        'reactive_power': 'Reactive Power Production Comparison',
        'voltage': 'Voltage Magnitude Comparison',
        'voltage_angle': 'Voltage Angle Comparison',
        'price': 'Nodal Price Comparison ($/MWh)'
    }[plotType];

    currentChart = new Chart(chartCanvas, {
        type: chartType,
        data: {
            labels: chartType === 'bar' ? sortedBusNumbers : undefined,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: 'category',
                    labels: sortedBusNumbers,
                    title: {
                        display: true,
                        text: 'Bus Number',
                        font: { weight: 'bold' }
                    },
                    ticks: {
                        autoSkip: false,
                        maxRotation: 45,
                        minRotation: 45,
                        font: { size: 11 }
                    }
                },
                y: {
                    beginAtZero: plotType !== 'voltage',
                    title: {
                        display: true,
                        text: yAxisTitle,
                        font: { weight: 'bold' }
                    },
                    ticks: {
                        callback: function(value) {
                            if (plotType === 'price') return '$' + value.toFixed(2);
                            if (plotType === 'reactive_power') return parseFloat(value.toFixed(3));
                            return value;
                        }
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: chartTitle,
                    font: { size: 16, weight: 'bold' },
                    padding: { top: 10, bottom: 20 }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw ?? context.parsed?.y ?? 'N/A';
                            const unit = {
                                'active_power': 'MW',
                                'reactive_power': 'MVAr',
                                'voltage': 'pu',
                                'voltage_angle': '°',
                                'price': '$/MWh'
                            }[plotType];

                            const formatted = value !== 'N/A'
                                ? (plotType === 'price' ? `$${value.toFixed(2)}` : value.toFixed(2))
                                : 'N/A';

                            return `${context.dataset.label}: ${formatted} ${unit}`;
                        }
                    }
                },
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 12,
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        font: { size: 12 }
                    }
                }
            }
        }
    });
}

// Helper functions
function getModelColor(modelType) {
    const colors = {
        'bolognani': '#cc2a36',
        'btheta': '#4f372d',
        'decoupled': '#00a0b0',
        'ac': '#edc951'
    };
    return colors[modelType] || `#${Math.floor(Math.random()*16777215).toString(16)}`;
}

function clearAll() {
    uploadedFiles = [];
    updateFileList();
    if (currentChart) {
        currentChart.destroy();
        currentChart = null;
    }
}

// Make functions available globally
window.removeFile = removeFile;