document.addEventListener("DOMContentLoaded", function () {
  // Validate username and email availability
  document.getElementById("username").addEventListener("blur", function() {
    if (checkUsernameLength(this.value, "usernameError")) {
        checkAvailability("username", this.value, "usernameError");
    }
  });
  document.getElementById("email").addEventListener("blur", function() {
      checkAvailability("email", this.value, "emailError");
  });

  // Validate password strength
  document.getElementById("password").addEventListener("input", function() {
      validatePasswordStrength(this.value, "passwordError");
  });

  // Validate password confirmation
  document.getElementById("confirm-password").addEventListener("input", function() {
      confirmPasswordMatch("password", "confirm-password", "confirmPasswordError");
  });
});

function checkUsernameLength(username, errorElementId) {
  let errorElement = document.getElementById(errorElementId);
  if (username.length < 4 || username.length > 12) {
      errorElement.textContent = 'Username must be between 4 and 12 characters long.';
      return false; // Indicates there is an error
  } else {
      errorElement.textContent = '';
      return true; // No error
  }
}

function checkAvailability(type, value, errorElementId) {
  if (!value) return; // Don't check if the value is empty
  fetch(`/check-availability?type=${type}&value=${value}`)
  .then(response => response.json())
  .then(data => {
      let errorElement = document.getElementById(errorElementId);
      if (data.exists) {
          errorElement.textContent = `${type.charAt(0).toUpperCase() + type.slice(1)} already taken.`;
      } else {
          errorElement.textContent = '';
      }
  })
  .catch(error => console.error('Error:', error));
}

function validatePasswordStrength(password, errorElementId) {
  let errorElement = document.getElementById(errorElementId);
  if (password.length < 8 || !/[a-z]/.test(password) || !/[A-Z]/.test(password) ||
      !/[0-9]/.test(password) || !/[_@$!%*?&]/.test(password)) {
      errorElement.textContent = 'Password must be at least 8 characters long and include lowercase, uppercase, numbers, and special characters.';
  } else {
      errorElement.textContent = '';
  }
}

function confirmPasswordMatch(passwordElementId, confirmPasswordElementId, errorElementId) {
  let password = document.getElementById(passwordElementId).value;
  let confirmPassword = document.getElementById(confirmPasswordElementId).value;
  let errorElement = document.getElementById(errorElementId);
  if (password !== confirmPassword) {
      errorElement.textContent = 'Passwords do not match.';
  } else {
      errorElement.textContent = '';
  }
}