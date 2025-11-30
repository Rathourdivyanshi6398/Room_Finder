// DOM Elements
const locationInput = document.getElementById('locationInput');
const cityInput = document.getElementById('cityInput'); // New city input element
const minPriceSlider = document.getElementById('minPriceSlider'); // Kept for UI label consistency
const maxPriceSlider = document.getElementById('maxPriceSlider');
const priceRangeLabel = document.getElementById('priceRangeLabel');
const searchButton = document.getElementById('searchButton');
const resultsTitle = document.getElementById('resultsTitle');
const roomsContainer = document.getElementById('roomsContainer');
const loadingIndicator = document.getElementById('loadingIndicator'); // Assume you have this element

// Backend API base URL
const API_BASE_URL = 'https://room-finder-dcz1.onrender.com'; // Ensure this matches your Flask app address

// --- State Variables ---
let currentPage = 1;
const roomsPerPage = 12; // Should match DEFAULT_PAGE_SIZE in Flask if possible
let isLoading = false;
let allRoomsLoaded = false;
let currentSearchLocation = '';
let currentSearchCity = ''; // Store the city used for the current search
let currentSearchBudget = 0; // Store the max budget used for the current search

let minPrice = 5000; // Default UI min value
let maxPrice = 25000; // Default UI max value

// Initialize the page
document.addEventListener('DOMContentLoaded', function () {
    // Set up price range sliders (primarily for UI display)
    minPriceSlider.addEventListener('input', updatePriceRange);
    maxPriceSlider.addEventListener('input', updatePriceRange);

    // Set initial values for sliders and label
    minPriceSlider.value = minPrice;
    maxPriceSlider.value = maxPrice;
    updatePriceRange(); // Update label initially

    // Set up search button
    searchButton.addEventListener('click', handleSearch);

    // Add Scroll Listener for Infinite Scrolling
    window.addEventListener('scroll', handleScroll);

    // Initial State Message
    resultsTitle.textContent = 'Find Your Perfect Room';
    roomsContainer.innerHTML = '<p class="initial-message">Please enter a location, city, and adjust the price range, then click Search.</p>';
});

// Update price range label when sliders change (UI only)
function updatePriceRange() {
    minPrice = parseInt(minPriceSlider.value);
    maxPrice = parseInt(maxPriceSlider.value);

    // Ensure min doesn't exceed max in the UI
    if (minPrice > maxPrice) {
        if (this.id === 'minPriceSlider') {
            minPriceSlider.value = maxPrice;
            minPrice = maxPrice;
        } else {
            maxPriceSlider.value = minPrice;
            maxPrice = minPrice;
        }
    }
    priceRangeLabel.textContent = `₹${formatNumber(minPrice)} - ₹${formatNumber(maxPrice)}`;
}

// Format number with commas for thousands (Indian Locale)
function formatNumber(num) {
    const number = Number(num);
    if (isNaN(number)) {
        return 'N/A';
    }
    // Using en-IN locale for Indian numbering style (lakhs, crores if large enough)
    return new Intl.NumberFormat('en-IN').format(number);
}

// --- Handle Scroll Event for Lazy Loading ---
function handleScroll() {
    const scrollThreshold = 300; // Pixels from bottom
    // Check if user is near the bottom, not currently loading, and more rooms might exist
    const userIsNearBottom = (window.innerHeight + window.scrollY) >= (document.body.offsetHeight - scrollThreshold);

    if (userIsNearBottom && !isLoading && !allRoomsLoaded) {
        console.log("Near bottom, loading more...");
        loadMoreRooms(currentSearchLocation, currentSearchCity, currentSearchBudget);
    }
}

// --- Handle search button click ---
function handleSearch() {
    const location = locationInput.value.trim();
    const city = cityInput.value.trim(); // Get city value from the new input
    const budget = maxPrice; // Use the maxPrice from the slider as the budget filter

    if (!location) {
        alert('Please enter a location to search.');
        return;
    }

    // Reset state for a new search
    currentPage = 1;
    isLoading = false;
    allRoomsLoaded = false;
    roomsContainer.innerHTML = ''; // Clear previous results
    
    // Update title with location and city if provided
    let searchDescription = `Searching for rooms near ${location}`;
    if (city) {
        searchDescription += ` in ${city}`;
    }
    resultsTitle.textContent = searchDescription;
    
    // Store current search parameters
    currentSearchLocation = location;
    currentSearchCity = city;
    currentSearchBudget = budget;

    loadMoreRooms(location, city, budget); // Start loading the first page
}

// --- Load More Rooms (handles API call and pagination) ---
async function loadMoreRooms(location, city, budget) {
    if (isLoading || allRoomsLoaded) {
        return; // Don't fetch if already loading or all loaded
    }

    isLoading = true;
    showLoadingIndicator();

    // Construct query parameters for the API
    const queryParams = new URLSearchParams({
        location: location,
        budget: budget, // API expects 'budget' (max price)
        page: currentPage,
        limit: roomsPerPage
    });
    
    // Add city parameter only if it's provided
    if (city) {
        queryParams.append('city', city);
    }
    
    const apiUrl = `${API_BASE_URL}/search?${queryParams.toString()}`;      

    console.log(`Fetching from: ${apiUrl}`);

    try {
        const response = await fetch(apiUrl);
        if (!response.ok) {
            let errorBody = 'Server returned an error.';
            try {
                // Try to get more specific error from response body
                const errorData = await response.json();
                errorBody = errorData.message || errorData.error || JSON.stringify(errorData);
            } catch (e) { /* Ignore if response body isn't valid JSON */ }
            throw new Error(`HTTP error! status: ${response.status} - ${errorBody}`);
        }

        const data = await response.json(); // Expect { results: [...], pagination: {...} }

        // Validate API response structure
        if (!data || typeof data !== 'object' || !Array.isArray(data.results) || !data.pagination) {
           console.error('API did not return expected format:', data);
           throw new Error('Unexpected API response format.');
        }

        const rooms = data.results;
        const paginationInfo = data.pagination;

        // Check if the results are still relevant (user might have started a new search)
        if (location !== currentSearchLocation || city !== currentSearchCity || budget !== currentSearchBudget) {
            console.log("Received results for a previous search, ignoring.");
            hideLoadingIndicator();
            isLoading = false;
            return; // Discard results from old search
        }

        if (rooms.length > 0) {
            appendRooms(rooms); // Add new rooms to the container
            currentPage++; // Increment page number for the next fetch
        }

        // Update title after the first page load
        if (paginationInfo.current_page === 1) {
             if (paginationInfo.total_results > 0) {
                 let resultsDescription = `${paginationInfo.total_results} Room${paginationInfo.total_results > 1 ? 's' : ''} found near ${location}`;
                 if (city) {
                     resultsDescription += ` in ${city}`;
                 }
                 resultsTitle.textContent = resultsDescription;
             } else {
                 let noResultsDescription = `No rooms found near ${location}`;
                 if (city) {
                     noResultsDescription += ` in ${city}`;
                 }
                 noResultsDescription += ` within budget ₹${formatNumber(budget)}`;
                 resultsTitle.textContent = noResultsDescription;
                 roomsContainer.innerHTML = '<p>No results match your criteria. Try adjusting the location, city, or budget.</p>';
             }
        }

        // Check if we've loaded all available rooms based on pagination info
        if (paginationInfo.current_page >= paginationInfo.total_pages) {
            allRoomsLoaded = true;
            if (paginationInfo.total_results > 0) { // Only show "no more" if there were results initially
                 showNoMoreRoomsMessage();
            }
        }

    } catch (error) {
        console.error('Error fetching rooms:', error);
        resultsTitle.textContent = 'Could not fetch rooms'; // Update title on error
        // Display error message in the container
        roomsContainer.innerHTML = `<p class="error-message">Failed to load rooms: ${error.message}. Please check the API server or try again later.</p>`;
        allRoomsLoaded = true; // Stop trying to load more on error

    } finally {
        hideLoadingIndicator();
        isLoading = false; // Release lock
    }
}

// --- Append rooms to the container ---
function appendRooms(rooms) {
    // Basic validation for required fields from the new API response
    rooms.forEach(room => {
        if (!room || typeof room !== 'object' || !room.title || room.price === undefined || !room.subtitle ) {
            console.warn('Skipping invalid room data (missing essential fields):', room);
            return; // Skip this room if essential data is missing
        }
        const card = createRoomCard(room);
        roomsContainer.appendChild(card);
    });
}

// --- Create a room card HTML element ---
function createRoomCard(room) {
    const card = document.createElement('div');
    card.className = 'room-card';

    const formattedPrice = formatNumber(room.price);
    const placeholderUrl = 'images/placeholder-image.jpg';
    const altText = room.title ? `Image for ${room.title}` : 'Room image';
    const imageUrl = room.first_image_url || placeholderUrl;

    let roomDetails = '';
    if (room.area && room.area !== 'Area Not Specified') {
        roomDetails += room.area;
    }
    if (room.property_details && room.property_details !== 'No Details Available') {
        roomDetails += (roomDetails ? ` - ${room.property_details}` : room.property_details);
    }
    if (!roomDetails) {
        roomDetails = 'Details not specified';
    }

    card.innerHTML = `
        <div class="room-card-image">
            <img src="${imageUrl}" 
                 alt="${altText}" 
                 loading="lazy" 
                 onerror="this.onerror=null; this.src='${placeholderUrl}';">
        </div>
        <div class="room-card-content">
            <div class="room-card-header">
                <h3 class="room-card-title">${room.title || 'N/A'}</h3>
                <p class="room-card-price">₹${formattedPrice}</p>
            </div>
            <div class="room-card-info">
                <!-- ... keep existing SVG and content ... -->
            </div>
            <div class="room-card-footer">
                <button class="btn btn-primary view-details-btn">View Details</button>
            </div>
        </div>
    `;

    // Add click handler for details button
    card.querySelector('.view-details-btn').addEventListener('click', async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/property-details?first_image_url=${encodeURIComponent(room.first_image_url)}`);
            const details = await response.json();
            
            // Create and show modal with details
            showDetailsModal(details);
        } catch (error) {
            console.error('Error fetching details:', error);
            alert('Could not load property details');
        }
    });

    return card;
}

// Add this modal display function
function showDetailsModal(details) {
    const modal = document.createElement('div');
    modal.className = 'details-modal';
    
    // Parse stringified data from backend
    const highlights = JSON.parse(details.highlights.replace(/'/g, '"'));
    const tableData = JSON.parse(details.table_data.replace(/'/g, '"'));

    modal.innerHTML = `
        <div class="modal-content">
            <h2>${details.location}</h2>
            <div class="modal-image">
                <img src="${details.first_image_url}" alt="Property image">
                <a href="${details.url}" target="_blank" class="original-listing">View Original Listing</a>
            </div>
            
            <div class="details-section">
                <h3>Highlights</h3>
                <ul>${highlights.map(h => `<li>${h}</li>`).join('')}</ul>
            </div>
            
            <div class="details-section">
                <h3>Property Specifications</h3>
                <div class="specs-grid">
                    ${Object.entries(tableData).map(([key, value]) => `
                        <div class="spec-item">
                            <span class="spec-key">${key}:</span>
                            <span class="spec-value">${value}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            <button class="close-modal">&times; Close</button>
        </div>
    `;

    // Close functionality
    modal.querySelector('.close-modal').addEventListener('click', () => modal.remove());
    modal.addEventListener('click', (e) => {
        if (e.target === modal) modal.remove();
    });

    document.body.appendChild(modal);
}


// --- Helper functions for Loading Indicator and "No More Rooms" message ---
function showLoadingIndicator() {
    // Use a dedicated element if it exists
    if (loadingIndicator) {
        loadingIndicator.style.display = 'block';
        return;
    }
    // Fallback: Append a temporary text element if no dedicated indicator found
    if (!roomsContainer.querySelector('.loading-text')) {
        const loadingText = document.createElement('p');
        loadingText.className = 'loading-text'; // Apply styles via CSS
        loadingText.textContent = 'Loading more rooms...';
        // Insert before the "no more rooms" message if it exists
        const noMoreMsg = roomsContainer.querySelector('.no-more-rooms');
        if (noMoreMsg) {
             roomsContainer.insertBefore(loadingText, noMoreMsg);
        } else {
             roomsContainer.appendChild(loadingText);
        }
    }
}

function hideLoadingIndicator() {
    if (loadingIndicator) {
        loadingIndicator.style.display = 'none';
    }
    // Remove the temporary text element if it was used
    const loadingText = roomsContainer.querySelector('.loading-text');
    if (loadingText) {
        loadingText.remove();
    }
}

function showNoMoreRoomsMessage() {
    // Prevent adding multiple messages
    if (!roomsContainer.querySelector('.no-more-rooms')) {
        const noMoreMessage = document.createElement('p');
        noMoreMessage.className = 'no-more-rooms'; // Apply styles via CSS
        noMoreMessage.textContent = 'You\'ve reached the end of the results.';
        roomsContainer.appendChild(noMoreMessage);
    }
}