document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const resultContainer = document.getElementById('result-container');
    const submitBtn = document.querySelector('.submit-btn');
    const resultTitle = document.getElementById('result-title');
    const resultText = document.getElementById('result-text');
    const probabilityBar = document.getElementById('probability-bar');
    const resetBtn = document.getElementById('reset-btn');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Form data collection
        const formData = new FormData(form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            if (value === '') {
                data[key] = null;
            } else if (['Pclass', 'Age', 'SibSp', 'Parch', 'Fare'].includes(key)) {
                data[key] = parseFloat(value);
            } else {
                data[key] = value;
            }
        }

        // UX: Show loading
        submitBtn.classList.add('loading');
        submitBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Prediction failed');
            }

            const result = await response.json();
            showResult(result);
        } catch (error) {
            console.error(error);
            alert('An error occurred while making the prediction. Please try again.');
        } finally {
            submitBtn.classList.remove('loading');
            submitBtn.disabled = false;
        }
    });

    function showResult(result) {
        form.classList.add('hidden');
        resultContainer.classList.remove('hidden');

        const probPercent = (result.probability * 100).toFixed(1);
        
        // Reset bar for animation
        probabilityBar.style.width = '0%';
        
        setTimeout(() => {
            probabilityBar.style.width = `${probPercent}%`;
            if (result.survived) {
                probabilityBar.style.background = 'var(--survived)';
                probabilityBar.style.boxShadow = '0 0 10px var(--survived)';
            } else {
                probabilityBar.style.background = 'var(--died)';
                probabilityBar.style.boxShadow = '0 0 10px var(--died)';
            }
        }, 100);

        if (result.survived) {
            resultTitle.textContent = 'Survived!';
            resultTitle.className = 'survived-text';
            resultText.innerHTML = `This passenger had a <strong class="survived-text">${probPercent}%</strong> chance of survival.`;
        } else {
            resultTitle.textContent = 'Did Not Survive';
            resultTitle.className = 'died-text';
            resultText.innerHTML = `This passenger had only a <strong class="survived-text">${probPercent}%</strong> chance of survival.`;
        }
    }

    resetBtn.addEventListener('click', () => {
        resultContainer.classList.add('hidden');
        form.classList.remove('hidden');
        probabilityBar.style.width = '0%';
    });
});
