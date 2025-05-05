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

function processVideo(videoFile) {
  const formData = new FormData();
  formData.append('video', videoFile);

  fetch('/video_classification', {
      method: 'POST',
      body: formData
  })
  .then(response => response.json())
  .then(data => {
      if (data.frame_results) {
          displayVideoClassificationResults(data.frame_results);
      }
  })
  .catch(error => {
      console.error('Error uploading video:', error);
  });
}

function uploadVideo() {
  const formData = new FormData(document.getElementById('uploadForm'));
  const loadingSpinner = document.getElementById('loadingSpinner');
  const resultContainer = document.getElementById('resultVideoContainer');

  loadingSpinner.classList.remove('hidden');
  resultContainer.innerHTML = '';

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
  const videoContainer = document.getElementById('resultVideoContainer');
  const spinner = document.getElementById('loadingSpinner');

  spinner.classList.add('hidden');

  videoContainer.innerHTML = '';

  const videoElement = document.createElement('video');
  videoElement.controls = true;
  videoElement.width = 640;
  videoElement.src = videoUrl;
  videoElement.classList.add('rounded-lg', 'shadow-lg');

  videoContainer.appendChild(videoElement);
}