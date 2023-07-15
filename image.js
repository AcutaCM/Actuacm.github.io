const showImageButton = document.querySelector('.show-image-button');
const imageContainer = document.querySelector('.image-container');
const image = imageContainer.querySelector('img');

showImageButton.addEventListener('click', () => {
  imageContainer.classList.add('active');
});

imageContainer.addEventListener('click', () => {
  imageContainer.classList.remove('active');
});