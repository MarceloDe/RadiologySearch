import React, { useState } from 'react';
import './ImageCarousel.css';

const ImageCarousel = ({ images, paperTitle }) => {
  const [currentIndex, setCurrentIndex] = useState(0);

  if (!images || images.length === 0) {
    return null;
  }

  const goToPrevious = () => {
    setCurrentIndex((prevIndex) => 
      prevIndex === 0 ? images.length - 1 : prevIndex - 1
    );
  };

  const goToNext = () => {
    setCurrentIndex((prevIndex) => 
      prevIndex === images.length - 1 ? 0 : prevIndex + 1
    );
  };

  const goToSlide = (index) => {
    setCurrentIndex(index);
  };

  const currentImage = images[currentIndex];

  return (
    <div className="image-carousel">
      <div className="carousel-header">
        <h5>Extracted Images from: {paperTitle}</h5>
        <span className="image-counter">{currentIndex + 1} / {images.length}</span>
      </div>
      
      <div className="carousel-container">
        <button 
          className="carousel-button prev" 
          onClick={goToPrevious}
          disabled={images.length <= 1}
        >
          ‹
        </button>
        
        <div className="carousel-content">
          <div className="image-wrapper">
            <img 
              src={currentImage.url} 
              alt={currentImage.alt_text || currentImage.caption}
              onError={(e) => {
                e.target.onerror = null;
                e.target.src = 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgdmlld0JveD0iMCAwIDQwMCAzMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxyZWN0IHdpZHRoPSI0MDAiIGhlaWdodD0iMzAwIiBmaWxsPSIjRjBGMEYwIi8+CjxwYXRoIGQ9Ik0xNTAgMTUwQzE1MCAxMjIuMzg2IDE3Mi4zODYgMTAwIDIwMCAxMDBDMjI3LjYxNCAxMDAgMjUwIDEyMi4zODYgMjUwIDE1MEMyNTAgMTc3LjYxNCAyMjcuNjE0IDIwMCAyMDAgMjAwQzE3Mi4zODYgMjAwIDE1MCAxNzcuNjE0IDE1MCAxNTBaIiBmaWxsPSIjRDBEMEQwIi8+CjxwYXRoIGQ9Ik0xODAgMTQwSDE5MFYxNjBIMTgwVjE0MFoiIGZpbGw9IiNBMEEwQTAiLz4KPHBhdGggZD0iTTIxMCAxNDBIMjIwVjE2MEgyMTBWMTQwWiIgZmlsbD0iI0EwQTBBMCIvPgo8L3N2Zz4=';
              }}
            />
          </div>
          
          <div className="image-details">
            {currentImage.figure_number && (
              <div className="figure-number">{currentImage.figure_number}</div>
            )}
            <div className="image-caption">{currentImage.caption}</div>
            <div className="relevance-score">
              Relevance: {(currentImage.relevance_score * 100).toFixed(0)}%
            </div>
          </div>
        </div>
        
        <button 
          className="carousel-button next" 
          onClick={goToNext}
          disabled={images.length <= 1}
        >
          ›
        </button>
      </div>
      
      {images.length > 1 && (
        <div className="carousel-indicators">
          {images.map((_, index) => (
            <button
              key={index}
              className={`indicator ${index === currentIndex ? 'active' : ''}`}
              onClick={() => goToSlide(index)}
              aria-label={`Go to image ${index + 1}`}
            />
          ))}
        </div>
      )}
    </div>
  );
};

export default ImageCarousel;