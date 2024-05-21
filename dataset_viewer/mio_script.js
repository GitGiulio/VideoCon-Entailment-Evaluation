let currentEntryIndex = 0;

        // Function to fetch data from CSV file
        async function fetchData(url) {
            try {
                const response = await fetch(url);
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                const data = await response.text();
                return data;
            } catch (error) {
                console.error('Error fetching data:', error);
                return null;
            }
        }

        // Function to parse CSV data
        function parseCSV(data) {
            return Papa.parse(data, { header: true }).data;
        }

        function parseCSVWithoutHeader(data) {
            // data = 'videopath,conversation,entailment\n' + data;
            return Papa.parse(data, { header: false }).data;
        }

        // Function to display current video
        function displayCurrentVideo(data, index) {
            const container = document.getElementById('videos-container');
            container.innerHTML = ''; // Clear previous content

            const entry = data[index];
            const caption = entry.caption;
            const videopathsMap = entry.videopaths;

            // Create container for the caption
            const captionContainer = document.createElement('div');
            captionContainer.classList.add('caption-container');
            const captionHeader = document.createElement('h2');
            captionHeader.textContent = caption;
            captionContainer.appendChild(captionHeader);
            container.appendChild(captionContainer);

            let maxMeanScore = -Infinity;
            let videoWithMaxScore = null;

            // Calculate maximum mean score across all videos
            for (const [videopath, entailmentScores] of videopathsMap) {
                const meanScore = entailmentScores.reduce((acc, [model, score]) => acc + parseFloat(score), 0) / entailmentScores.length;
                if (meanScore > maxMeanScore) {
                    maxMeanScore = meanScore;
                    videoWithMaxScore = videopath;
                }
            }

            let maxScores = new Map();
            for (const [videopath, entailmentScores] of videopathsMap) {
                for (const [model, score] of entailmentScores) {
                    if (!maxScores.has(model)) {
                        maxScores.set(model, [-Infinity, null]);
                    }
                    if (parseFloat(score) > maxScores.get(model)[0]) {
                        maxScores.set(model, [parseFloat(score), videopath]);
                    }
                }
            }

            // Create container for videos, organize videos in rows of 5
            let counter = 0; // Counter to limit 5 videos per row
            let videoRowContainer = document.createElement('div');
            videoRowContainer.classList.add('video-row');
            for (const [videopath, entailmentScores] of videopathsMap) {
                const meanScore = entailmentScores.reduce((acc, [model, score]) => acc + parseFloat(score), 0) / entailmentScores.length;

                const videoContainer = document.createElement('div');
                videoContainer.classList.add('video-container');
                const videoElement = document.createElement('video');
                videoElement.autoplay = true;
                videoElement.controls = true;
                videoElement.loop = true;
                videoElement.muted = true;
                videoElement.src = videopath;

                videoContainer.appendChild(videoElement);
                // const scoreText = document.createElement('p');
                if (entailmentScores.length > 0) {
                    // iterate over entailment scores and display them each with the corresponding model in its own paragraph
                    for (const [model, score] of entailmentScores) {
                        const scoreText = document.createElement('p');
                        scoreText.textContent = `${model}: ${Number(score).toFixed(2)}`;

                        if (maxScores.has(model) && videopath === maxScores.get(model)[1]) {
                            scoreText.classList.add(`${model.replace(/\./g, '')}-highlighted-video`);
                        }

                        videoContainer.appendChild(scoreText);
                    }
                    // const formattedScores = entailmentScores.map(score => Number(score).toFixed(2));
                    // scoreText.textContent = `Entailment: ${formattedScores.join(', ')}`;
                } else {
                    scoreText.textContent = 'Entailment: N/A'; // No entailment score available
                }
                const meanScoreText = document.createElement('p');
                meanScoreText.textContent = `Mean Entailment: ${meanScore.toFixed(2)}`;
                if (videopath === videoWithMaxScore) {
                    meanScoreText.classList.add('mean-highlighted-video');
                }
                // videoContainer.appendChild(scoreText);
                videoContainer.appendChild(meanScoreText);
                videoRowContainer.appendChild(videoContainer);

                // if (videopath === videoWithMaxScore) {
                //     videoContainer.classList.add('highlighted-video');
                // }

                counter++;
                if (counter === 5) {
                    container.appendChild(videoRowContainer); // Append completed row to the container
                    videoRowContainer = document.createElement('div'); // Create new row container
                    videoRowContainer.classList.add('video-row');
                    counter = 0; // Reset counter for the next row
                }
            }
            // Append any remaining videos if the total number of videos is not a multiple of 5
            if (counter > 0) {
                container.appendChild(videoRowContainer);
            }
        }

        // Function to handle next button click
        function onNextButtonClick(data) {
            currentEntryIndex = (currentEntryIndex + 1) % data.length;
            displayCurrentVideo(data, currentEntryIndex);
        }

        // Main function
        async function main() {
            const response = await fetch('../data/complete_df.csv');
            const data = await response.text();
            let parsedData = parseCSV(data);


            // Initialize videosData array
            let videosData = [];

            // Perform first iteration over uniqueCaptions
            const firstCaption = uniqueCaptions.shift(); // Remove and return the first caption
            await processCaption(firstCaption, videosData);

            // Display the first set of videos
            displayCurrentVideo(videosData, currentEntryIndex);

            const nextButton = document.getElementById('next-button');
            nextButton.addEventListener('click', async () => {
                let videosData = [];
                if (parsedData.length > 0) {
                    const nextRow = parsedData.shift(); // Remove and return the next caption
                    await processCaption(nextRow, videosData);
                    // Display the next set of videos
                    displayCurrentVideo(videosData, currentEntryIndex);
                }
            });
        }

        // Function to process a caption and fetch its video paths
        async function processRow(row, framesData) {
            try {
                const videopathsMap = new Map(); // Use a Set to store unique videopaths
            } catch (error) {
                console.error('Error processing caption:', error);
            }
        }

        main();