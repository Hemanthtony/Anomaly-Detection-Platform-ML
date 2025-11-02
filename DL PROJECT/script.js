document.getElementById('upload-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const fileInput = document.getElementById('image-input');
    const resultDiv = document.getElementById('result');

    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (response.ok) {
                const plantName = data.plant_name;
                const probability = (data.probability * 100).toFixed(2);
                const commonNames = data.common_names ? data.common_names.join(', ') : 'N/A';
                const healthStatus = data.health_status;
                const healthProbability = (data.health_probability * 100).toFixed(2);
                const isHealthy = healthStatus.toLowerCase().includes('healthy');

                resultDiv.style.display = 'block';
                resultDiv.style.backgroundColor = isHealthy ? '#e8f5e8' : '#ffebee';
                resultDiv.style.border = `1px solid ${isHealthy ? '#4CAF50' : '#f44336'}`;
                resultDiv.innerHTML = `<p>Image "${file.name}" uploaded successfully!</p><p>Plant identified: ${plantName}</p><p>Probability: ${probability}%</p><p>Common names: ${commonNames}</p><p>Health status: ${healthStatus} (${healthProbability}%)</p>`;
            } else {
                resultDiv.style.display = 'block';
                resultDiv.style.backgroundColor = '#ffebee';
                resultDiv.style.border = '1px solid #f44336';
                resultDiv.innerHTML = `<p>Error: ${data.error}</p>`;
            }
        } catch (error) {
            resultDiv.style.display = 'block';
            resultDiv.style.backgroundColor = '#ffebee';
            resultDiv.style.border = '1px solid #f44336';
            resultDiv.innerHTML = '<p>Failed to connect to the server. Please try again.</p>';
        }
    } else {
        resultDiv.style.display = 'block';
        resultDiv.style.backgroundColor = '#ffebee';
        resultDiv.style.border = '1px solid #f44336';
        resultDiv.innerHTML = '<p>Please select an image file.</p>';
    }
});
