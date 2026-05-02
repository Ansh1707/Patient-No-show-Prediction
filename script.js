const apiUrl = "http://localhost:8000/predict"; // Update if your backend runs elsewhere

document.getElementById("predictBtn").addEventListener("click", async () => {
  const form = document.getElementById("predictionForm");
  const formData = new FormData(form);
  const data = {};

  formData.forEach((value, key) => {
    // Convert numeric fields to numbers
    const numFields = [
  "Age", "wait_days", "Scholarship",
  "Hipertension", "Diabetes", "Alcoholism",
  "Handcap", "SMS_received"
];
    data[key] = numFields.includes(key) ? Number(value) : value;
  });

  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data)
    });

    if (!response.ok) throw new Error(`Error: ${response.statusText}`);
    const result = await response.json();

    const resultDiv = document.getElementById("result");
    resultDiv.classList.remove("d-none", "alert-danger", "alert-success");

    if (result.prediction === 1) {
      resultDiv.classList.add("alert-danger");
      resultDiv.textContent = `🚨 High risk of no-show! Probability: ${result.probability_no_show}`;
    } else {
      resultDiv.classList.add("alert-success");
      resultDiv.textContent = `✅ Patient likely to attend. Probability of no-show: ${result.probability_no_show}`;
    }
  } catch (err) {
    alert("Failed to get prediction: " + err.message);
  }
});
