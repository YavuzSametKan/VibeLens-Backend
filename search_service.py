import requests
import os
import time
import random
import re  # YENİ: URL ve Başlık temizliği için Regex
from io import BytesIO
from PIL import Image
import hashlib
from dotenv import load_dotenv
from models import Category
from duckduckgo_search import DDGS

load_dotenv()
TMDB_KEY = os.getenv("TMDB_API_KEY")

PLACEHOLDER_IMG = "https://placehold.co/600x900?text=No+Image"


def clean_query_for_api(title: str) -> str:
    """
    'Kurtarıcı Projesi (Project Hail Mary)' gibi karmaşık başlıkları temizler.
    Open Library gibi hassas API'ler için gereklidir.
    """
    # 1. Parantez içindeki metni al (Genelde orijinal isim ordadır)
    match = re.search(r'\((.*?)\)', title)
    if match:
        return match.group(1)  # "Project Hail Mary" döner

    # Parantez yoksa olduğu gibi döndür
    return title


def is_valid_image(url: str) -> bool:
    """
    Resmin boyutunu ve bütünlüğünü kontrol eder.
    """
    if not url or "placehold.co" in url: return False

    # Google Books için ön kontrol (Hızlı eleme)
    if "books.google.com" in url and "zoom=0" not in url:
        # Eğer zoom=0 değilse düzeltmeyi denemediysek şüpheli
        pass

    try:
        response = requests.get(url, timeout=4)
        if response.status_code != 200: return False

        img_data = response.content
        img = Image.open(BytesIO(img_data))

        # 1. PİKSEL KONTROLÜ:
        if img.width < 50 or img.height < 50:
            print(f"⚠️ Resim çok küçük ({img.width}x{img.height}). Reddedildi.")
            return False

        # 2. DOSYA BOYUTU KONTROLÜ:
        if len(img_data) < 2500:  # 2.5KB altı kesinlikle placeholderdır
            print(f"⚠️ Dosya boyutu şüpheli ({len(img_data)} bytes). Reddedildi.")
            return False

        return True
    except Exception:
        return False


def search_image_fallback(query: str) -> str:
    """
    Web Scraping (DuckDuckGo). Son çare.
    """
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


# --- OPEN LIBRARY (YEDEK) ---
def _fetch_from_open_library(title: str, creator: str) -> str:
    """
    Open Library Covers API.
    Başlık temizliği yaparak arama şansını artırır.
    """
    # Başlığı temizle: "Kurtarıcı Projesi (Project Hail Mary)" -> "Project Hail Mary"
    cleaned_title = clean_query_for_api(title)

    print(f"🏛️ Open Library Sorgusu: '{cleaned_title} {creator}'")

    search_url = "https://openlibrary.org/search.json"
    params = {"q": f"{cleaned_title} {creator}", "limit": 1}

    try:
        res = requests.get(search_url, params=params, timeout=5).json()
        if res.get('docs'):
            doc = res['docs'][0]
            # cover_i varsa kullan
            if doc.get('cover_i'):
                cover_id = doc['cover_i']
                return f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
            # ISBN varsa onu dene (daha garanti)
            elif doc.get('isbn'):
                isbn = doc['isbn'][0]
                return f"https://covers.openlibrary.org/b/isbn/{isbn}-L.jpg"
    except Exception as e:
        print(f"OpenLib Hatası: {e}")
    return None


def get_poster_url(title: str, creator: str, category: Category) -> str:
    image_url = None

    try:
        if category == Category.MOVIE:
            image_url = _fetch_from_tmdb(title, "movie")
        elif category == Category.SERIES:
            image_url = _fetch_from_tmdb(title, "tv")
        elif category == Category.MUSIC:
            image_url = _fetch_from_itunes_music(f"{title} {creator}")

        elif category == Category.BOOK:
            # 1. Google Books (En İyi Eşleşme)
            print(f"📖 Google Books deneniyor: {title}")
            image_url = _fetch_from_google_books(title)

            # 2. Validasyon: Google patlarsa -> Open Library Dene
            if not image_url or not is_valid_image(image_url):
                print(f"🏛️ Open Library deneniyor (Google başarısız): {title}")
                image_url = _fetch_from_open_library(title, creator)

    except Exception as e:
        print(f"API Hatası: {e}")

    # 3. Validasyon: Hala yoksa -> Scraping
    if not image_url or not is_valid_image(image_url):
        print(f"🔄 API'ler başarısız. Scraping devreye giriyor...")
        scrape_query = f"{title} {creator} {category.value} official cover high resolution"
        image_url = search_image_fallback(scrape_query)

    return image_url if image_url else PLACEHOLDER_IMG


# --- API FONKSİYONLARI ---

def _fetch_from_tmdb(query: str, type: str) -> str:
    if not TMDB_KEY: return None
    url = f"https://api.themoviedb.org/3/search/{type}"
    params = {"api_key": TMDB_KEY, "query": query, "language": "tr-TR"}
    try:
        res = requests.get(url, params=params).json()
        if res.get('results') and res['results'][0].get('poster_path'):
            return f"https://image.tmdb.org/t/p/w500{res['results'][0]['poster_path']}"
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
            # Thumbnail'i al ama üzerinde oynayacağız
            best = links.get('extraLarge') or links.get('large') or links.get('medium') or links.get('thumbnail')

            if best:
                # 1. HTTP -> HTTPS
                best = best.replace("http://", "https://")

                # 2. ZOOM FIX (Regex ile kesin çözüm)
                # &zoom=1 veya &zoom=2 ne varsa bulup &zoom=0 yapıyoruz
                best = re.sub(r'&zoom=\d', '&zoom=0', best)

                # 3. Gereksiz parametre temizliği
                best = best.replace("&edge=curl", "")

                return best
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