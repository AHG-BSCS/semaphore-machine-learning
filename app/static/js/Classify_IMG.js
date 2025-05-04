    document.addEventListener("DOMContentLoaded", function () {
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
            })
            .catch(error => {
                console.error("Error fetching models:", error);
            });
    });

    function uploadImage() {
    const formData = new FormData();
    const fileInput = document.getElementById('image-upload');
    const modelSelect = document.getElementById('model-select');
    const selectedModel = modelSelect.value;

    if (!fileInput.files[0]) {
        alert("Please select an image.");
        return;
    }

    if (!selectedModel) {
        alert("Please select a model.");
        return;
    }

    formData.append("image", fileInput.files[0]);
    formData.append("model_name", selectedModel);

    const resultContainer = document.getElementById('resultImageContainer');
    resultContainer.innerHTML = `<div class="loading-spinner"></div>`;

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/object-detection/", true);
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                const response = JSON.parse(xhr.responseText);
                resultContainer.innerHTML = `
                    <div class="relative-container">
                        <div class="image-container">
                            <img src="data:image/png;base64,${response.result_img}" alt="Processed Image" class="border border-gray-300 rounded-lg shadow dark:border-gray-700 max-w-md">
                        </div>
                        <div class="detection-text">${response.detected_text || "No detection"}</div>
                    </div>
                `;
                setTimeout(() => {
                    window.scrollTo({
                        top: document.body.scrollHeight,
                        behavior: 'smooth'
                    });
                }, 100);
            } else {
                resultContainer.innerHTML = `<p class="text-red-500">Error: ${xhr.responseText}</p>`;
            }
        }
    };

    xhr.send(formData);
    document.getElementById('image-upload').value = '';
    }