async function uploadImage() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    if (!file) {
        alert("请先选择图片！");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    await fetch("/api/upload/", {
        method: "POST",  // 这里使用POST请求
        body: formData
    });

    const response = await fetch("/api/infer/");  // 使用GET请求
    const data = await response.json();

    document.getElementById("resultText").textContent = JSON.stringify(data.results, null, 2);
    document.getElementById("resultImage").src = "data:image/png;base64," + data.image_base64;
}
