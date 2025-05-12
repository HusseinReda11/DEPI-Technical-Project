/**
 * Client-side JavaScript for the sentiment analysis application
 */

/**
 * Analyze the sentiment of the text in the tweet input field
 */
async function analyzeSentiment() {
    // Get the input element
    const tweetInput = document.getElementById('tweet-input');
    const resultDiv = document.getElementById('result');
    const tweet = tweetInput.value;

    // Validate input
    if (tweet.trim() === "") {
        showError("Please enter a tweet to analyze.");
        return;
    }

    // Show loading indicator
    resultDiv.innerHTML = '<div class="loading">Analyzing...</div>';

    try {
        // Send the request to the API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ tweets: [{ text: tweet }] })
        });

        // Handle HTTP errors
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        // Parse the response
        const data = await response.json();
        
        // Check for errors in the response
        if (data.length === 0 || data[0].error) {
            throw new Error(data[0]?.error || "Unknown error in response");
        }

        // Get the results from the first tweet
        const result = data[0];
        const sentiment = result.sentiment;
        const confidence = result.confidence;

        // Display the result
        const sentimentClass = sentiment === "positive" ? "positive" : "negative";
        resultDiv.innerHTML = `
            <div class="result-container">
                <div class="sentiment ${sentimentClass}">
                    ${sentiment.toUpperCase()}
                </div>
                <div class="confidence">
                    Confidence: ${(confidence * 100).toFixed(1)}%
                </div>
            </div>
        `;

    } catch (error) {
        showError(error.message);
        console.error('Error during sentiment analysis:', error);
    }
}

/**
 * Display an error message to the user
 * @param {string} message - The error message to display
 */
function showError(message) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `<div class="error">Error: ${message}</div>`;
}

/**
 * Initialize the application when the DOM is loaded
 */
document.addEventListener('DOMContentLoaded', function() {
    // Add event listener for the analyze button
    const analyzeButton = document.getElementById('analyze-button');
    if (analyzeButton) {
        analyzeButton.addEventListener('click', analyzeSentiment);
    }

    // Add event listener for pressing Enter in the textarea
    const tweetInput = document.getElementById('tweet-input');
    if (tweetInput) {
        tweetInput.addEventListener('keypress', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                analyzeSentiment();
            }
        });
    }
});