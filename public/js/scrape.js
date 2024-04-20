document.addEventListener("DOMContentLoaded", function() {
  const form = document.querySelector('.scrape-form');
  const scrapeUrl = form.getAttribute('data-scrape-url'); // Get the URL from the data attribute

  form.addEventListener('submit', function(e) {
      e.preventDefault();

      const articleUrl = document.getElementById('article_url').value;
      const data = { 'article_url': articleUrl };

      fetch(scrapeUrl, { // Use the URL from the data attribute
          method: 'POST',
          headers: {
              'Content-Type': 'application/x-www-form-urlencoded'
          },
          body: new URLSearchParams(data)
      })
      .then(response => response.text())
      .then(html => {
          const parser = new DOMParser();
          const doc = parser.parseFromString(html, 'text/html');
          const flashMessages = doc.querySelector('.flash-messages');
          const summary = doc.querySelector('.summary-wrapper p').textContent;

          if (flashMessages) {
              const flashContainer = document.querySelector('.flash-messages');
              if (flashContainer) {
                  flashContainer.innerHTML = flashMessages.innerHTML;
              } else {
                  form.prepend(flashMessages);
              }
          }

          // Update the summary section
          document.querySelector('.summary-wrapper p').textContent = summary;
      })
      .catch((error) => {
          console.error('Error:', error);
      });
  });
});
