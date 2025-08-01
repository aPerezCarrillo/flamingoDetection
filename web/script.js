document.addEventListener('DOMContentLoaded', () => {
    const videoPlayer = document.getElementById('video-player');
    const chartCanvas = document.getElementById('timeline-chart');

    const CLASS_COLORS = {
        0: 'rgba(255, 105, 180, 0.7)', // Flamingo (Pink)
        1: 'rgba(135, 206, 250, 0.7)'  // Human (Sky Blue)
    };

    // --- Chart.js plugin to draw the vertical line ---
    const verticalLinePlugin = {
        id: 'verticalLine',
        currentFrame: 0, // A property to hold the current frame
        beforeDatasetsDraw(chart, args, options) {
            const { ctx, chartArea: { top, bottom }, scales: { x } } = chart;
            const xCoord = x.getPixelForValue(this.currentFrame);

            if (xCoord >= x.left && xCoord <= x.right) {
                ctx.save();
                ctx.beginPath();
                ctx.lineWidth = 2;
                ctx.strokeStyle = 'rgba(255, 0, 0, 0.7)'; // Red line
                ctx.moveTo(xCoord, top);
                ctx.lineTo(xCoord, bottom);
                ctx.stroke();
                ctx.restore();
            }
        }
    };

    fetch('timeline.json')
        .then(response => response.json())
        .then(data => {
            const timelineData = data.timeline;
            const videoMetadata = data.video_metadata;

            const labels = [];
            const chartData = [];
            const backgroundColors = [];
            const sortedTrackIDs = Object.keys(timelineData).sort((a, b) => parseInt(a) - parseInt(b));

            for (const trackId of sortedTrackIDs) {
                const track = timelineData[trackId];
                labels.push(track.display_id); // Use the new display_id
                chartData.push([track.first_seen, track.last_seen]);
                backgroundColors.push(CLASS_COLORS[track.class_id] || 'rgba(200, 200, 200, 0.7)');
            }

            const chart = new Chart(chartCanvas, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Object Presence by Frame',
                        data: chartData,
                        backgroundColor: backgroundColors,
                        borderColor: backgroundColors.map(c => c.replace('0.7', '1')),
                        borderWidth: 1,
                        barPercentage: 0.8,
                        categoryPercentage: 0.9
                    }]
                },
                plugins: [verticalLinePlugin], // Register the custom plugin
                options: {
                    indexAxis: 'y',
                    scales: {
                        x: { min: 0, max: videoMetadata.total_frames, title: { display: true, text: 'Frame Number' } }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `Present from frame ${context.raw[0]} to ${context.raw[1]}.`;
                                }
                            }
                        }
                    },
                    onClick: (event, elements) => {
                        if (elements.length > 0) {
                            const trackId = sortedTrackIDs[elements[0].index];
                            const startTimeSeconds = timelineData[trackId].first_seen / videoMetadata.fps;
                            videoPlayer.currentTime = startTimeSeconds;
                            //videoPlayer.play();
                        }
                    }
                }
            });

            // --- Sync video time to chart ---
            videoPlayer.addEventListener('timeupdate', () => {
                const currentFrame = Math.floor(videoPlayer.currentTime * videoMetadata.fps);
                verticalLinePlugin.currentFrame = currentFrame; // Update the plugin's state
                chart.update('none'); // Redraw the chart without animation
            });
        })
        .catch(error => console.error('Error loading timeline data:', error));
});