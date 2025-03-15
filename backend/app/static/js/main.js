// Common error handling
function handleError(error, elementId = 'errorMessage') {
    const errorDiv = document.getElementById(elementId);
    if (errorDiv) {
        errorDiv.textContent = error.message || 'An error occurred';
        errorDiv.classList.remove('d-none');
    } else {
        console.error(error);
    }
}

// Show/hide loading spinner
function toggleSpinner(spinnerId, show) {
    const spinner = document.getElementById(spinnerId);
    if (spinner) {
        spinner.classList.toggle('d-none', !show);
    }
}

// Format number for display
function formatNumber(number) {
    if (typeof number === 'number') {
        return number.toLocaleString(undefined, {
            maximumFractionDigits: 4
        });
    }
    return number;
}

// Create table from data
function createTable(data, columns) {
    const table = document.createElement('table');
    table.className = 'table table-striped';
    
    // Create header
    const thead = document.createElement('thead');
    const headerRow = document.createElement('tr');
    columns.forEach(column => {
        const th = document.createElement('th');
        th.textContent = column;
        headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);
    table.appendChild(thead);
    
    // Create body
    const tbody = document.createElement('tbody');
    data.forEach(row => {
        const tr = document.createElement('tr');
        columns.forEach(column => {
            const td = document.createElement('td');
            td.textContent = formatNumber(row[column]);
            tr.appendChild(td);
        });
        tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    
    return table;
}

// Handle API requests
async function apiRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            ...options,
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            }
        });
        
        if (!response.ok) {
            throw new Error(`API request failed: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        handleError(error);
        throw error;
    }
} 