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

plotBtn.addEventListener('click', generatePlot);
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
                } else {
                    alert(`Cannot determine model type from filename: ${file.name}\nPlease include 'btheta', 'bolognani', or 'decoupled' in the filename.`);
                    return;
                }

                // Verify the file has a Production sheet
                if (workbook.SheetNames.some(name => name.toLowerCase() === 'production')) {
                    uploadedFiles.push({
                        name: file.name,
                        model: modelType,
                        data: workbook
                    });
                    updateFileList();
                } else {
                    alert(`File ${file.name} doesn't contain active power data (missing Production sheet)`);
                }
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

document.getElementById('fileInput').addEventListener('change', function () {
  const fileList = Array.from(this.files).map(file => file.name).join(', ');
  document.getElementById('fileNameDisplay').textContent = fileList || "No files selected.";
});


// Plot generation
// Global variables for plot control
let plotDataExtents = {
    x: { min: 0, max: 0, range: 0, median: 0 },
    y: { min: 0, max: 0, range: 0, median: 0 }
};
let currentZoomLevel = 100;
let isManualYRange = false;






function generatePlot(plotType = 'active_power') {
    if (uploadedFiles.length === 0) {
        alert('Please upload at least one result file');
        return;
    }

    const isVoltagePlot = plotType === 'voltage';
    const isAnglePlot = plotType === 'voltage_angle';
    const isPricePlot = plotType === 'price';
    const isPowerPlot = !isVoltagePlot && !isAnglePlot && !isPricePlot;

    // Collect all unique bus numbers as strings from the appropriate sheets
    const busNumbers = new Set();
    const datasets = [];
    const allYValues = [];

    // First pass: collect buses and prepare datasets
    uploadedFiles.forEach(file => {
        let sheetName, busColumn, valueColumn;
        
        if (isPowerPlot) {
            if (plotType === 'reactive_power' && file.model === 'bolognani') {
                sheetName = 'Reactive_Production';
                valueColumn = 'q_pu';
            } else {
                sheetName = 'Production';
                valueColumn = plotType === 'active_power' ? 'production' : 'q';
            }
        } 
        else if (isPricePlot) {
            sheetName = 'Price';
            valueColumn = file.model === 'bolognani' ? 'price' : 'node_price';
        }
        else {
            sheetName = 'Results';
            if (file.model === 'bolognani') {
                valueColumn = isVoltagePlot ? 'vm_pu' : 'va_degree';
            } else {
                valueColumn = isVoltagePlot ? 'V_pu' : 'Delta';
            }
        }

        const sheet = file.data.Sheets[sheetName];
        if (!sheet) return;

        const data = XLSX.utils.sheet_to_json(sheet);
        busColumn = data.some(row => 'bus' in row) ? 'bus' : 'Bus';

        // Collect bus numbers as strings
        data.forEach(row => {
            if (row[busColumn]) {
                busNumbers.add(row[busColumn].toString());
            }
        });
    });

    // Convert to sorted array of bus strings
    const sortedBusNumbers = Array.from(busNumbers).sort((a, b) => parseInt(a) - parseInt(b));

    if (sortedBusNumbers.length === 0) {
        alert(`No ${plotType} data found in uploaded files`);
        return;
    }

    // Second pass: create datasets with model-specific styling
    uploadedFiles.forEach(file => {
        let sheetName, busColumn, valueColumn;
        
        if (isPowerPlot) {
            if (plotType === 'reactive_power' && file.model === 'bolognani') {
                sheetName = 'Reactive_Production';
                valueColumn = 'q_pu';
            } else {
                sheetName = 'Production';
                valueColumn = plotType === 'active_power' ? 'production' : 'q';
            }
        }
        else if (isPricePlot) {
            sheetName = 'Price';
            valueColumn = file.model === 'bolognani' ? 'price' : 'node_price';
        }
        else {
            sheetName = 'Results';
            if (file.model === 'bolognani') {
                valueColumn = isVoltagePlot ? 'vm_pu' : 'va_degree';
            } else {
                valueColumn = isVoltagePlot ? 'V_pu' : 'Delta';
            }
        }

        const sheet = file.data.Sheets[sheetName];
        if (!sheet) return;

        const data = XLSX.utils.sheet_to_json(sheet);
        busColumn = data.some(row => 'bus' in row) ? 'bus' : 'Bus';

        const datasetConfig = {
            label: `${file.model.charAt(0).toUpperCase() + file.model.slice(1)}`,
            backgroundColor: getModelColor(file.model),
            borderColor: getModelColor(file.model),
            borderWidth: 2
        };

        if (isVoltagePlot || isAnglePlot || isPricePlot) {
            const dataPoints = [];
            data.forEach(row => {
                if (row[busColumn] && row[valueColumn] !== undefined) {
                    const value = isPricePlot 
                        ? parseFloat(parseFloat(row[valueColumn]).toFixed(2))
                        : row[valueColumn];
                    
                    dataPoints.push({
                        x: row[busColumn].toString(), // Keep as string
                        y: value
                    });
                }
            });

            // Sort by bus number
            dataPoints.sort((a, b) => parseInt(a.x) - parseInt(b.x));

            // Model-specific point styling
            Object.assign(datasetConfig, {
                data: dataPoints,
                pointBackgroundColor: getModelColor(file.model),
                pointRadius: file.model === 'bolognani' ? 5 : 4,          // Larger for Bolognani
                pointHoverRadius: file.model === 'bolognani' ? 7 : 6,
                pointStyle: file.model === 'btheta' ? 'rect' : 'circle',    // Square for BTheta
                borderDash: file.model === 'decoupled' ? [5, 5] : [],      // Dashed line for Decoupled
                showLine: true,
                lineTension: 0,
                fill: false,
                borderWidth: file.model === 'bolognani' ? 2 : 1            // Thicker line for Bolognani
            });
        } else {
            const fileDataMap = {};
            data.forEach(row => {
                if (row[busColumn] && row[valueColumn] !== undefined) {
                    fileDataMap[row[busColumn].toString()] = row[valueColumn];
                }
            });

            Object.assign(datasetConfig, {
                data: sortedBusNumbers.map(bus => fileDataMap[bus] || null),
                borderWidth: 1,
                borderRadius: file.model === 'bolognani' ? 3 : 0           // Rounded corners for Bolognani bars
            });
        }

        datasets.push(datasetConfig);
    });

    if (currentChart) {
        currentChart.destroy();
    }

    // Create chart with proper bus numbers on x-axis
    const chartConfig = {
        type: (isVoltagePlot || isAnglePlot || isPricePlot) ? 'scatter' : 'bar',
        data: {
            labels: isPowerPlot ? sortedBusNumbers : undefined,
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
                        font: {
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        autoSkip: false,
                        maxRotation: 45,
                        minRotation: 45,
                        font: {
                            size: 11
                        }
                    }
                },
                y: {
                    beginAtZero: isPowerPlot,
                    title: {
                        display: true,
                        text: plotType === 'active_power' ? 'Active Power (MW)' : 
                              plotType === 'reactive_power' ? 'Reactive Power (MVAr)' :
                              plotType === 'voltage' ? 'Voltage Magnitude (pu)' :
                              plotType === 'voltage_angle' ? 'Voltage Angle (degrees)' :
                              'Price ($/MWh)',
                        font: {
                            weight: 'bold'
                        }
                    },
                    ticks: {
                        callback: function(value) {
                            if (isPricePlot) {
                                return '$' + value.toFixed(2); // Format prices with dollar sign
                            }
                            // Add special formatting for reactive power
                            if (plotType === 'reactive_power') {
                                return parseFloat(value.toFixed(3)); // 3 decimal places for reactive power
                            }
                            return value;
                        }
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: plotType === 'active_power' ? 'Active Power Production Comparison' : 
                          plotType === 'reactive_power' ? 'Reactive Power Production Comparison' :
                          plotType === 'voltage' ? 'Voltage Magnitude Comparison' :
                          plotType === 'voltage_angle' ? 'Voltage Angle Comparison' :
                          'Nodal Price Comparison ($/MWh)',
                    font: {
                        size: 16,
                        weight: 'bold'
                    },
                    padding: {
                        top: 10,
                        bottom: 20
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const value = context.raw !== null ? context.raw : 
                                       (context.parsed ? context.parsed.y : null);
                            const unit = plotType === 'active_power' ? 'MW' : 
                                     plotType === 'reactive_power' ? 'MVAr' :
                                     plotType === 'voltage' ? 'pu' :
                                     plotType === 'voltage_angle' ? '°' :
                                     '$/MWh';
                            const formattedValue = value !== null ? 
                                (isPricePlot ? '$' + value.toFixed(2) : value.toFixed(2)) : 
                                'N/A';
                            return `${context.dataset.label}: ${formattedValue} ${unit}`;
                        }
                    },
                    displayColors: true,
                    usePointStyle: true,
                    padding: 12,
                    bodySpacing: 8
                },
                legend: {
                    position: 'top',
                    labels: {
                        boxWidth: 12,
                        padding: 20,
                        usePointStyle: true,
                        pointStyle: 'circle',
                        font: {
                            size: 12
                        }
                    }
                }
            },
            elements: {
                line: {
                    tension: 0 // Straight lines between points
                }
            }
        }
    };

    currentChart = new Chart(chartCanvas, chartConfig);
}

























// Helper functions for plot controls
function calculateDataExtents(datasets, allBuses) {
    // Get all y values
    const allYValues = datasets.flatMap(dataset => 
        dataset.data.map(point => point ? point.y : null)
        .filter(val => val !== null)
    );
    
    // Get all x values (bus numbers)
    const allXValues = allBuses.map(Number);
    
    // Calculate extents
    plotDataExtents = {
        x: {
            min: Math.min(...allXValues),
            max: Math.max(...allXValues),
            range: Math.max(...allXValues) - Math.min(...allXValues),
            median: (Math.min(...allXValues) + Math.max(...allXValues)) / 2
        },
        y: {
            min: Math.min(...allYValues),
            max: Math.max(...allYValues),
            range: Math.max(...allYValues) - Math.min(...allYValues),
            median: (Math.min(...allYValues) + Math.max(...allYValues)) / 2
        }
    };
    
    // Update placeholder values
    document.getElementById('yMin').placeholder = plotDataExtents.y.min.toFixed(2);
    document.getElementById('yMax').placeholder = plotDataExtents.y.max.toFixed(2);
}

function resetPlotControls() {
    currentZoomLevel = 100;
    isManualYRange = false;
    document.getElementById('zoomSlider').value = currentZoomLevel;
    document.getElementById('zoomValue').textContent = currentZoomLevel + '%';
    document.getElementById('yMin').value = '';
    document.getElementById('yMax').value = '';
}

function updateChartZoom() {
    if (!currentChart || !plotDataExtents) return;
    
    const zoomFactor = currentZoomLevel / 100;
    const effectiveRange = Math.max(plotDataExtents.x.range, plotDataExtents.y.range) * zoomFactor;
    
    // Update X-axis range (always auto-scaled)
    currentChart.options.scales.x.min = plotDataExtents.x.median - effectiveRange/2;
    currentChart.options.scales.x.max = plotDataExtents.x.median + effectiveRange/2;
    
    // Update Y-axis range (manual or auto)
    if (isManualYRange) {
        const yMin = parseFloat(document.getElementById('yMin').value);
        const yMax = parseFloat(document.getElementById('yMax').value);
        if (!isNaN(yMin) && !isNaN(yMax) && yMax > yMin) {
            currentChart.options.scales.y.min = yMin;
            currentChart.options.scales.y.max = yMax;
        }
    } else {
        currentChart.options.scales.y.min = plotDataExtents.y.median - effectiveRange/2;
        currentChart.options.scales.y.max = plotDataExtents.y.median + effectiveRange/2;
    }
    
    currentChart.update();
}

// Initialize controls when page loads
document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('zoomSlider').addEventListener('input', function() {
        currentZoomLevel = parseInt(this.value);
        document.getElementById('zoomValue').textContent = currentZoomLevel + '%';
        updateChartZoom();
    });

    document.getElementById('applyYRange').addEventListener('click', function() {
        const yMin = parseFloat(document.getElementById('yMin').value);
        const yMax = parseFloat(document.getElementById('yMax').value);
        
        if (!isNaN(yMin) && !isNaN(yMax) && yMax > yMin) {
            isManualYRange = true;
            updateChartZoom();
        } else {
            alert('Please enter valid Y-axis range (min < max)');
        }
    });

    document.getElementById('resetYRange').addEventListener('click', function() {
        document.getElementById('yMin').value = '';
        document.getElementById('yMax').value = '';
        isManualYRange = false;
        updateChartZoom();
    });

    document.getElementById('resetZoom').addEventListener('click', function() {
        currentZoomLevel = 100;
        document.getElementById('zoomSlider').value = currentZoomLevel;
        document.getElementById('zoomValue').textContent = currentZoomLevel + '%';
        updateChartZoom();
    });
});



// Update event listener
plotBtn.addEventListener('click', () => {
    const plotType = document.getElementById('plotType').value;
    generatePlot(plotType);
});

// Data extraction
function extractActivePower(workbook, modelType) {
    try {
        const sheet = workbook.Sheets['Production'];
        if (!sheet) {
            console.warn(`No Production sheet found in ${modelType} file`);
            return null;
        }

        const data = XLSX.utils.sheet_to_json(sheet);
        
        // Handle both 'bus' and 'Bus' column names
        const busColumn = data.some(row => 'bus' in row) ? 'bus' : 'Bus';
        
        return data.map(row => ({
            bus: row[busColumn].toString(),  // Use detected column name
            value: row.production
        })).filter(item => item.value !== undefined);
    } catch (e) {
        console.error(`Error extracting active power from ${modelType}:`, e);
        return null;
    }
}


function extractReactivePower(workbook, modelType) {
    try {
        // BTheta model should be ignored for reactive power
        if (modelType === 'btheta') {
            return null;
        }

        let sheet, data, busColumn, valueColumn;
        
        if (modelType === 'bolognani') {
            sheet = workbook.Sheets['Reactive_Production'];
            if (!sheet) return null;
            data = XLSX.utils.sheet_to_json(sheet);
            busColumn = data.some(row => 'Bus' in row) ? 'Bus' : 'bus';
            valueColumn = 'q_pu';
        } 
        else if (modelType === 'decoupled') {
            sheet = workbook.Sheets['Production'];
            if (!sheet) return null;
            data = XLSX.utils.sheet_to_json(sheet);
            busColumn = data.some(row => 'Bus' in row) ? 'Bus' : 'bus';
            valueColumn = 'q';
        }

        if (!data || !busColumn || !valueColumn) return null;

        return data.map(row => ({
            bus: row[busColumn].toString(),
            value: row[valueColumn]
        })).filter(item => item.value !== undefined);

    } catch (e) {
        console.error(`Error extracting reactive power from ${modelType}:`, e);
        return null;
    }
}

function extractVoltageMagnitude(workbook, modelType) {
    console.log(`Extracting voltage for ${modelType} model`); // Debug log
    
    try {
        // Check if workbook has 'Results' sheet
        const sheetNames = workbook.SheetNames.map(name => name.toLowerCase());
        if (!sheetNames.includes('results')) {
            console.warn(`No 'Results' sheet found. Available sheets:`, workbook.SheetNames);
            return null;
        }

        const sheet = workbook.Sheets['Results'];
        const data = XLSX.utils.sheet_to_json(sheet);
        console.log('Raw sheet data:', data.slice(0, 3)); // Log first 3 rows for inspection

        // Determine column names based on model type
        let busColumn, voltageColumn;
        switch(modelType.toLowerCase()) {
            case 'bolognani':
                busColumn = data.some(row => 'bus' in row) ? 'bus' : 'Bus';
                voltageColumn = 'vm_pu';
                break;
            case 'btheta':
            case 'decoupled':
                busColumn = data.some(row => 'Bus' in row) ? 'Bus' : 'bus';
                voltageColumn = 'V_pu';
                break;
            default:
                console.warn(`Unknown model type: ${modelType}`);
                return null;
        }

        // Verify columns exist
        if (!data[0] || !(busColumn in data[0]) || !(voltageColumn in data[0])) {
            console.warn(`Required columns not found. Available columns:`, Object.keys(data[0]));
            return null;
        }

        // Extract and filter data
        const result = data.map(row => ({
            bus: row[busColumn].toString(),
            value: parseFloat(row[voltageColumn])
        })).filter(item => !isNaN(item.value));

        console.log(`Extracted ${result.length} voltage points`);
        return result.length > 0 ? result : null;

    } catch (e) {
        console.error(`Error extracting voltage from ${modelType}:`, e);
        return null;
    }
}

function extractPrice(workbook, modelType) {
    try {
        const sheet = workbook.Sheets['Price'];
        if (!sheet) {
            console.warn(`No Price sheet found in ${modelType} file`);
            return null;
        }

        const data = XLSX.utils.sheet_to_json(sheet);

        // Handle different column names for different models
        let priceColumn;
        switch(modelType) {
            case 'bolognani':
                priceColumn = 'price';
                break;
            case 'decoupled':
            case 'btheta':
                priceColumn = 'node_price';
                break;
            default:
                return null;
        }

        // Handle both 'bus' and 'Bus' column names
        const busColumn = data.some(row => 'bus' in row) ? 'bus' : 'Bus';

        return data.map(row => ({
            bus: row[busColumn].toString(),
            value: row[priceColumn] !== undefined
                ? parseFloat(parseFloat(row[priceColumn]).toFixed(2))
                : undefined
        })).filter(item => item.value !== undefined);
    } catch (e) {
        console.error(`Error extracting nodal prices from ${modelType}:`, e);
        return null;
    }
}


function extractVoltageAngle(workbook, modelType) {
    try {
        const sheet = workbook.Sheets['Results'];
        if (!sheet) {
            console.warn(`No Results sheet found in ${modelType} file`);
            return null;
        }

        const data = XLSX.utils.sheet_to_json(sheet);
        
        // Handle different column names for different models
        let angleColumn;
        switch(modelType) {
            case 'bolognani':
                angleColumn = 'va_degree';
                break;
            case 'decoupled':
            case 'btheta':
                angleColumn = 'Delta';
                break;
            default:
                return null;
        }

        // Handle both 'bus' and 'Bus' column names
        const busColumn = data.some(row => 'bus' in row) ? 'bus' : 'Bus';
        
        return data.map(row => ({
            bus: row[busColumn].toString(),
            value: row[angleColumn]
        })).filter(item => item.value !== undefined);
    } catch (e) {
        console.error(`Error extracting voltage angle from ${modelType}:`, e);
        return null;
    }
}

// Helper functions
function getModelColor(modelType) {
    const colors = {
        'bolognani': '#4361ee',
        'btheta': '#4cc9f0',
        'decoupled': '#f72585'
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

// Make removeFile available globally
window.removeFile = removeFile;