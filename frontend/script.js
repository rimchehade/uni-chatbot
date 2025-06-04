const backendURL = "http://localhost:5000"; // Replace this with deployed backend if needed
let mediaRecorder;
let audioChunks = [];

async function submitText() {
  const question = document.getElementById("question").value;
  const res = await fetch(`${backendURL}/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });

  const data = await res.json();
  document.getElementById("response").innerText =
    `Answer: ${data.answer}\n\nFollow-up: ${data.follow_up}`;
}

async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  audioChunks = [];

  mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
  mediaRecorder.onstop = async () => {
    const blob = new Blob(audioChunks, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append("audio", blob, "recording.webm");

    const res = await fetch(`${backendURL}/audio`, {
      method: "POST",
      body: formData
    });

    const data = await res.json();
    document.getElementById("response").innerText =
      `Transcribed: ${data.question}\n\nAnswer: ${data.answer}\n\nFollow-up: ${data.follow_up}`;
  };

  mediaRecorder.start();
  document.getElementById("recording-status").innerText = "Recording...";
}

function stopRecording() {
  if (mediaRecorder) {
    mediaRecorder.stop();
    document.getElementById("recording-status").innerText = "Processing...";
  }
}
