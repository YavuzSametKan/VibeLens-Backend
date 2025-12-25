import requests
import time
import random
import re
import urllib.parse
from io import BytesIO
from PIL import Image
from duckduckgo_search import DDGS
from app.core.config import settings
from app.schemas.analysis import Category

# --- CONFIGURATION ---
PLACEHOLDER_IMG = "https://placehold.co/600x900?text=No+Image"
TMDB_KEY = settings.TMDB_API_KEY
FALLBACK_TIMEOUT_MIN = 1.5
FALLBACK_TIMEOUT_MAX = 3.0


# --- UTILITY HELPERS ---
def generate_music_links(artist: str, track: str, apple_url: str = None) -> dict:
    """Generates standard music service links for a track."""
    query = f"{artist} {track}"
    safe_query = urllib.parse.quote(query)

    links = {
        "spotify": f"https://open.spotify.com/search/{safe_query}",
        "youtube_music": f"https://music.youtube.com/search?q={safe_query}",
        "youtube": f"https://www.youtube.com/results?search_query={safe_query}"
    }

    if apple_url:
        links["apple_music"] = apple_url

    return links


def clean_query_for_api(title: str) -> str:
    """Removes parenthetical content and cleans the title for API queries."""
    cleaned = re.sub(r"\(.*?\)", "", title).strip()
    return cleaned


def is_valid_image(url: str) -> bool:
    """Checks if a URL points to a valid, substantial image."""
    if not url or "placehold.co" in url:
        return False
    
    # Trust TMDB URLs without validation (they're from a reliable API)
    if "image.tmdb.org" in url or "themoviedb.org" in url:
        print(f"✓ Trusted TMDB poster URL: {url}")
        return True
    
    # Specific Google Books check
    if "books.google.com" in url and "zoom=0" not in url:
        pass

    try:
        response = requests.get(url, timeout=4)
        if response.status_code != 200:
            print(f"⚠️ Image validation failed (status {response.status_code}): {url}")
            return False
        img_data = response.content
        img = Image.open(BytesIO(img_data))

        # Minimum size and data length checks
        if img.width < 50 or img.height < 50:
            print(f"⚠️ Image too small ({img.width}x{img.height}): {url}")
            return False
        if len(img_data) < 2500:
            print(f"⚠️ Image file too small ({len(img_data)} bytes): {url}")
            return False
        return True
    except Exception as e:
        print(f"⚠️ Image validation error: {e} - {url}")
        return False


def search_image_fallback(query: str) -> str:
    """Uses DuckDuckGo Search to find an image when APIs fail."""
    try:
        # Avoid rapid scraping
        time.sleep(random.uniform(FALLBACK_TIMEOUT_MIN, FALLBACK_TIMEOUT_MAX))
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=1, safesearch="off"))
            if results:
                return results[0]['image']
    except Exception:
        pass
    return PLACEHOLDER_IMG


# --- LOW-LEVEL API FETCHERS ---
def _fetch_tmdb_metadata(query: str, content_type: str) -> dict | None:
    """Fetches metadata for movies or TV series from TMDB."""
    if not TMDB_KEY:
        print("⚠️ TMDB API Key not configured")
        return None

    clean_query = clean_query_for_api(query)
    url = f"https://api.themoviedb.org/3/search/{content_type}"
    params = {"api_key": TMDB_KEY, "query": clean_query, "language": "tr-TR"}

    try:
        res = requests.get(url, params=params, timeout=5).json()
        results = res.get('results', [])
        if not results:
            print(f"⚠️ TMDB: No results found for '{query}' (cleaned: '{clean_query}')")
            return None

        # Select the best match based on vote count
        best_match = max(results, key=lambda x: x.get('vote_count', 0))
        print(f"✓ TMDB found: {best_match.get('title') or best_match.get('name', 'Unknown')} (votes: {best_match.get('vote_count', 0)})")

        # Poster URL construction
        poster = PLACEHOLDER_IMG
        if best_match.get('poster_path'):
            poster = f"https://image.tmdb.org/t/p/w500{best_match['poster_path']}"
            print(f"✓ Poster path found: {best_match['poster_path']}")
        else:
            print(f"⚠️ No poster_path in TMDB response for '{query}'")

        # Extract year (only if valid)
        date_field = 'release_date' if content_type == 'movie' else 'first_air_date'
        year_str = best_match.get(date_field, "")
        year = year_str[:4] if year_str and len(year_str) >= 4 else None

        # Extract overview (handle empty strings)
        overview = best_match.get('overview', "").strip()
        if not overview:
            overview = None  # Return None instead of empty string, so Gemini's value can be used
        elif len(overview) > 350:
            # Truncate if too long
            last_dot = overview[:350].rfind('.')
            if last_dot != -1:
                overview = overview[:last_dot + 1]
            else:
                overview = overview[:350] + "..."

        # Extract rating
        vote_average = best_match.get('vote_average', 0)
        rating = f"{vote_average:.1f}/10" if vote_average > 0 else None

        return {
            "poster": poster,
            "overview": overview,
            "rating": rating,
            "year": year
        }
    except Exception as e:
        print(f"⚠️ TMDB API Error for '{query}': {e}")
        return None


def _fetch_itunes_full_metadata(query: str) -> dict | None:
    """Fetches full music metadata (links, artwork) from iTunes."""
    url = "https://itunes.apple.com/search"
    params = {"term": query, "media": "music", "limit": 1}
    try:
        res = requests.get(url, params=params).json()
        if res['resultCount'] > 0:
            item = res['results'][0]
            # Replace 100x100 artwork with high-res 600x600
            artwork = item.get('artworkUrl100', '').replace('100x100', '600x600')
            artist = item.get('artistName', '')
            track = item.get('trackName', '')
            apple_link = item.get('trackViewUrl')
            links = generate_music_links(artist, track, apple_link)
            return {
                "poster": artwork,
                "external_links": links,
                "overview": None,
                "rating": None,
                "year": None
            }
    except:
        pass
    return None


def _fetch_book_poster_google(query: str) -> str | None:
    """Fetches book cover URL from Google Books API."""
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": query, "maxResults": 1}
    try:
        res = requests.get(url, params=params).json()
        if 'items' in res:
            links = res['items'][0]['volumeInfo'].get('imageLinks', {})
            best = links.get('extraLarge') or links.get('large') or links.get('medium') or links.get('thumbnail')
            if best:
                # Cleanup and standardization
                best = best.replace("http://", "https://")
                best = re.sub(r'&zoom=\d', '&zoom=0', best)
                return best.replace("&edge=curl", "")
    except:
        pass
    return None


def _fetch_book_poster_openlibrary(title: str, creator: str) -> str | None:
    """Fetches book cover URL from Open Library API."""
    cleaned_title = clean_query_for_api(title)
    search_url = "https://openlibrary.org/search.json"
    params = {"q": f"{cleaned_title} {creator}", "limit": 1}
    try:
        res = requests.get(search_url, params=params, timeout=5).json()
        if res.get('docs') and res['docs'][0].get('cover_i'):
            return f"https://covers.openlibrary.org/b/id/{res['docs'][0]['cover_i']}-L.jpg"
    except:
        pass
    return None


def _fetch_music_poster_itunes(query: str) -> str | None:
    """Fetches music artwork URL from iTunes API (poster only)."""
    url = "https://itunes.apple.com/search"
    params = {"term": query, "media": "music", "limit": 1}
    try:
        res = requests.get(url, params=params).json()
        if res['resultCount'] > 0:
            # High-res version of artwork
            return res['results'][0].get('artworkUrl100', '').replace('100x100bb', '600x600bb')
    except:
        pass
    return None


# --- POSTER RESOLVER ---
def get_poster_url(title: str, creator: str, category: Category) -> str:
    """Attempts to find the best poster URL using API fallbacks and scraping."""
    image_url = None

    try:
        if category == Category.BOOK:
            # 1. Try Google Books
            image_url = _fetch_book_poster_google(title)
            # 2. Try Open Library if Google fails or image is invalid
            if not image_url or not is_valid_image(image_url):
                image_url = _fetch_book_poster_openlibrary(title, creator)
        elif category == Category.MUSIC:
            image_url = _fetch_music_poster_itunes(f"{title} {creator}")
    except Exception:
        pass

    # Fallback to web scraping if API failed or image is invalid/not substantial
    if not image_url or not is_valid_image(image_url):
        scrape_query = f"{title} {creator} {category.value} official cover high resolution"
        image_url = search_image_fallback(scrape_query)

    return image_url if image_url else PLACEHOLDER_IMG


# --- MAIN FUNCTION: METADATA COLLECTOR ---
def get_content_metadata(title: str, creator: str, category: Category) -> dict:
    """Collects comprehensive metadata for a piece of content."""
    print(f"\n🔍 Fetching metadata for: '{title}' ({category.value})")
    
    metadata = {
        "poster": PLACEHOLDER_IMG,
        "overview": None,
        "rating": None,
        "year": None,
        "external_links": None
    }

    try:
        if category == Category.MOVIE:
            api_data = _fetch_tmdb_metadata(title, "movie")
            if api_data:
                print(f"✓ TMDB data received: poster={api_data.get('poster', 'N/A')[:50]}...")
                metadata.update(api_data)
            else:
                print(f"⚠️ TMDB returned no data for movie: '{title}'")

        elif category == Category.SERIES:
            api_data = _fetch_tmdb_metadata(title, "tv")
            if api_data:
                print(f"✓ TMDB data received: poster={api_data.get('poster', 'N/A')[:50]}...")
                metadata.update(api_data)
            else:
                print(f"⚠️ TMDB returned no data for series: '{title}'")

        elif category == Category.MUSIC:
            itunes_data = _fetch_itunes_full_metadata(f"{title} {creator}")
            if itunes_data:
                metadata.update(itunes_data)
            else:
                # Fetch poster and basic links if full iTunes data is missing
                metadata["poster"] = get_poster_url(title, creator, category)
                metadata["external_links"] = generate_music_links(creator, title)

        elif category == Category.BOOK:
            poster_url = get_poster_url(title, creator, category)
            metadata["poster"] = poster_url

        # Final Poster Fallback Check (For Movie/Series where TMDB failed)
        if category in [Category.MOVIE, Category.SERIES]:
            print(f"🖼️  Poster validation check for '{title}':")
            print(f"   Current poster: {metadata['poster']}")
            
            if not is_valid_image(metadata["poster"]):
                print(f"⚠️ Poster validation failed, attempting fallback scraping...")
                scrape_query = f"{title} {creator} {category.value} official poster"
                fallback_poster = search_image_fallback(scrape_query)
                metadata["poster"] = fallback_poster
                print(f"   Fallback poster: {fallback_poster}")
            else:
                print(f"✓ Poster validated successfully")

    except Exception as e:
        print(f"❌ Error in get_content_metadata for '{title}': {e}")
        # Catch any high-level errors and return default metadata
        pass

    print(f"📦 Final metadata poster: {metadata['poster']}\n")
    return metadata