const hamburger = document.querySelector('.hamburger-menu');
const navBar = document.querySelector('.main-nav');

hamburger.addEventListener('click', function() {
    navBar.classList.toggle('responsive');
});
