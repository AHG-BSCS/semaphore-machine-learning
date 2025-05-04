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