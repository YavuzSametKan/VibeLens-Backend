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

PLACEHOLDER_IMG = "https://placehold.co/600x900?text=No+Image"
TMDB_KEY = settings.TMDB_API_KEY


# --- YARDIMCI ARAÇLAR ---

def generate_music_links(artist: str, track: str, apple_url: str = None) -> dict:
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
    cleaned = re.sub(r"\(.*?\)", "", title).strip()
    return cleaned


def is_valid_image(url: str) -> bool:
    if not url or "placehold.co" in url: return False
    if "books.google.com" in url and "zoom=0" not in url: pass

    try:
        response = requests.get(url, timeout=4)
        if response.status_code != 200: return False
        img_data = response.content
        img = Image.open(BytesIO(img_data))

        if img.width < 50 or img.height < 50: return False
        if len(img_data) < 2500: return False
        return True
    except Exception:
        return False


def search_image_fallback(query: str) -> str:
    try:
        print(f"🕷️ Fallback: Web Scraping (DDG) deneniyor: '{query}'")
        time.sleep(random.uniform(1.5, 3.0))
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=1, safesearch="off"))
            if results:
                found = results[0]['image']
                print(f"✅ Scraping Başarılı: {found}")
                return found
    except Exception as e:
        print(f"⚠️ Scraping Hatası: {e}")
    return PLACEHOLDER_IMG


# --- ALT SEVİYE API FONKSİYONLARI ---

def _fetch_tmdb_metadata(query: str, type: str) -> dict:
    if not TMDB_KEY: return None

    clean_query = clean_query_for_api(query)
    url = f"https://api.themoviedb.org/3/search/{type}"
    params = {"api_key": TMDB_KEY, "query": clean_query, "language": "tr-TR"}

    try:
        res = requests.get(url, params=params).json()
        results = res.get('results', [])
        if not results: return None

        best_match = max(results, key=lambda x: x.get('vote_count', 0))
        poster = PLACEHOLDER_IMG
        if best_match.get('poster_path'):
            poster = f"https://image.tmdb.org/t/p/w500{best_match['poster_path']}"

        date_field = 'release_date' if type == 'movie' else 'first_air_date'
        year = best_match.get(date_field, "")[:4]

        overview = best_match.get('overview', "Özet yok.")
        if len(overview) > 350:
            last_dot = overview[:350].rfind('.')
            if last_dot != -1:
                overview = overview[:last_dot + 1]
            else:
                overview = overview[:350] + "..."

        return {
            "poster": poster,
            "overview": overview,
            "rating": f"{best_match.get('vote_average', 0):.1f}/10",
            "year": year
        }
    except:
        pass
    return None


def _fetch_itunes_full_metadata(query: str) -> dict:
    url = "https://itunes.apple.com/search"
    params = {"term": query, "media": "music", "limit": 1}
    try:
        res = requests.get(url, params=params).json()
        if res['resultCount'] > 0:
            item = res['results'][0]
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


def _fetch_from_google_books(query: str) -> str:
    url = "https://www.googleapis.com/books/v1/volumes"
    params = {"q": query, "maxResults": 1}
    try:
        res = requests.get(url, params=params).json()
        if 'items' in res:
            links = res['items'][0]['volumeInfo'].get('imageLinks', {})
            best = links.get('extraLarge') or links.get('large') or links.get('medium') or links.get('thumbnail')
            if best:
                best = best.replace("http://", "https://")
                best = re.sub(r'&zoom=\d', '&zoom=0', best)
                return best.replace("&edge=curl", "")
    except:
        pass
    return None


def _fetch_from_itunes_music(query: str) -> str:
    url = "https://itunes.apple.com/search"
    params = {"term": query, "media": "music", "limit": 1}
    try:
        res = requests.get(url, params=params).json()
        if res['resultCount'] > 0:
            return res['results'][0].get('artworkUrl100', '').replace('100x100bb', '600x600bb')
    except:
        pass
    return None


def _fetch_from_open_library(title: str, creator: str) -> str:
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


# --- POSTER BULUCU ---

def get_poster_url(title: str, creator: str, category: Category) -> str:
    image_url = None
    try:
        if category == Category.BOOK:
            image_url = _fetch_from_google_books(title)
            if not image_url or not is_valid_image(image_url):
                image_url = _fetch_from_open_library(title, creator)
        elif category == Category.MUSIC:
            image_url = _fetch_from_itunes_music(f"{title} {creator}")
    except Exception as e:
        print(f"API Hatası: {e}")

    if not image_url or not is_valid_image(image_url):
        scrape_query = f"{title} {creator} {category.value} official cover high resolution"
        image_url = search_image_fallback(scrape_query)

    return image_url if image_url else PLACEHOLDER_IMG


# --- ANA FONKSİYON: METADATA TOPLAYICI ---

def get_content_metadata(title: str, creator: str, category: Category) -> dict:
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
            if api_data: metadata.update(api_data)

        elif category == Category.SERIES:
            api_data = _fetch_tmdb_metadata(title, "tv")
            if api_data: metadata.update(api_data)

        elif category == Category.MUSIC:
            itunes_data = _fetch_itunes_full_metadata(f"{title} {creator}")
            if itunes_data:
                metadata.update(itunes_data)
            else:
                metadata["poster"] = get_poster_url(title, creator, category)
                metadata["external_links"] = generate_music_links(creator, title)

        elif category == Category.BOOK:
            poster_url = get_poster_url(title, creator, category)
            metadata["poster"] = poster_url

        # Yedekleme (Film/Dizi Poster Scraping)
        if category in [Category.MOVIE, Category.SERIES]:
            if not is_valid_image(metadata["poster"]):
                print(f"🔄 TMDB posteri yok. Scraping devreye giriyor...")
                scrape_query = f"{title} {creator} {category.value} official poster"
                metadata["poster"] = search_image_fallback(scrape_query)

    except Exception as e:
        print(f"Metadata Hatası: {e}")

    return metadata