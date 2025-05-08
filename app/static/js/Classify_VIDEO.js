fetch("/get-models")
  .then(response => response.json())
  .then(models => {
    const select = document.getElementById("model-select");
    models.forEach(model => {
      const option = document.createElement("option");
      option.value = model;
      option.textContent = model;
      if (model === "yolov12.pt") {
        option.selected = true;
      }
      select.appendChild(option);
    });
  });

document.getElementById('model-select').addEventListener('change', function () {
  fetch('/set_model', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model_name: this.value })
  })
  .then(response => response.json())
  .then(data => console.log(data.message || data.error));
});

function uploadVideo() {
  const formData = new FormData(document.getElementById('uploadForm'));
  const loadingSpinner = document.getElementById('loadingSpinner');
  const placeholder = document.getElementById('video-placeholder');
  const video = document.getElementById('processed-video');
  const source = document.getElementById('processed-video-source');

  document.getElementById('video-upload').value = '';

  video.classList.add('hidden');
  video.pause();
  source.src = '';
  video.load();

  placeholder.classList.add('hidden');
  loadingSpinner.classList.remove('hidden');

  startProgressBar();

  fetch('/video_classification', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    if (data.video_url) {
        displayProcessedVideo(data.video_url);
    } else {
        showError('An error occurred while processing the video.');
    }
  })
  .catch(error => {
    console.error(error);
    showError('An error occurred while processing the video.');
  });
}

function displayProcessedVideo(videoUrl) {
  const spinner = document.getElementById('loadingSpinner');
  const video = document.getElementById('processed-video');
  const source = document.getElementById('processed-video-source');

  spinner.classList.add('hidden');

  source.src = `${videoUrl}?t=${new Date().getTime()}`;
  video.load();
  video.classList.remove('hidden');

  setTimeout(() => {
    video.scrollIntoView({ behavior: 'smooth' });
  }, 750);

  video.play();
}

function startProgressBar() {
  const loadingSpinner = document.getElementById('loadingSpinner');
  const progressBar = document.getElementById('progress-bar');
  const progressText = document.getElementById('progress-percent');

  const interval = setInterval(() => {
    fetch('/video_progress')
      .then(response => response.json())
      .then(data => {
        const progress = data.progress;
        progressBar.value = progress;
        progressText.textContent = `${progress}%`;

        if (progress >= 100) {
          clearInterval(interval);
          setTimeout(() => {
            loadingSpinner.classList.add('hidden');
          }, 500);
        }
      });
  }, 1000);

  return interval;
}