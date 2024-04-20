document.addEventListener("DOMContentLoaded", function () {
  const loginForm = document.querySelector(".login-form");

  loginForm.addEventListener("submit", function(event) {
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    let isValid = true;

    // Clear previous error messages
    document.querySelectorAll('.error-message').forEach(e => e.textContent = '');

    // Check if username or password is empty
    if (!username) {
      displayError('username', 'Username is required.');
      isValid = false;
    }

    if (!password) {
      displayError('password', 'Password is required.');
      isValid = false;
    }

    // Prevent form submission if validation fails
    if (!isValid) {
      event.preventDefault();
    }
  });

  function displayError(fieldId, message) {
    const errorElementId = fieldId + "Error";
    const errorElement = document.getElementById(errorElementId) || createErrorElement(fieldId, errorElementId);
    errorElement.textContent = message;
  }

  function createErrorElement(fieldId, errorElementId) {
    const field = document.getElementById(fieldId);
    const errorElement = document.createElement("div");
    errorElement.id = errorElementId;
    errorElement.className = 'error-message';
    field.parentNode.insertBefore(errorElement, field.nextSibling);
    return errorElement;
  }
});

function resendVerification() {
  const username = document.getElementById('username').value;
  fetch('/resend-verification', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify({ username: username })
  })
  .then(response => response.json())
  .then(data => {
      alert(data.message);  // Alert the response message
  })
  .catch(error => console.error('Error:', error));
}

