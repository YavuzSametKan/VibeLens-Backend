Aşağıda, sağladığın `README.md` dosyasının, teknik terminolojiye sadık kalınarak hazırlanmış Türkçe çevirisi yer almaktadır.

---

# VibeLens Backend API

## Genel Bakış

VibeLens, yüz duygu analizine dayalı olarak kişiselleştirilmiş içerik önerileri (Filmler, Diziler, Kitaplar ve Müzik) sunmak için tasarlanmış gelişmiş bir arka uç servisidir. Bilgisayarlı Görü (Computer Vision - CV) ile Üretken Yapay Zekayı (Generative AI - LLM) birleştiren sistem; kullanıcının yüz ifadesini analiz eder, baskın ve ikincil duygu durumlarını belirler ve bu özel ruh haline göre uyarlanmış "klişeleşmemiş" kültürel önerilerden oluşan bir liste oluşturur.

Sistem, monolitik bir FastAPI uygulaması içerisinde mikroservis tabanlı bir mimariye dayanır; derin öğrenme çıkarımları (inference) için PyTorch'u, anlamsal akıl yürütme ve içerik kürasyonu için ise Google Gemini'ı kullanır.

## Canlı Analiz Örneği

![VibeLens Canlı Duygu Analizi](assets/vibelens_live_analysis.jpeg)

* **Karmaşık Duygu Mantığı:** İkincil duyguları ve ağırlıklı puanları hesaplayarak basit "Mutlu/Üzgün" sınıflandırmasının ötesine geçer.
* **Yapılandırılmış YZ Çıktısı:** Güvenilir ayrıştırma (parsing) sağlamak için LLM'den gelen çıktıları JSON şemasına zorlar...

## İçindekiler

1. [Sistem Mimarisi](https://www.google.com/search?q=%23sistem-mimarisi)
2. [Temel Özellikler](https://www.google.com/search?q=%23temel-%C3%B6zellikler)
3. [Teknoloji Yığını](https://www.google.com/search?q=%23teknoloji-y%C4%B1%C4%9F%C4%B1n%C4%B1)
4. [Ön Koşullar](https://www.google.com/search?q=%23%C3%B6n-ko%C5%9Fullar)
5. [Kurulum](https://www.google.com/search?q=%23kurulum)
6. [Yapılandırma](https://www.google.com/search?q=%23yap%C4%B1land%C4%B1rma)
7. [Kullanım](https://www.google.com/search?q=%23kullan%C4%B1m)
8. [Proje Yapısı](https://www.google.com/search?q=%23proje-yap%C4%B1s%C4%B1)
9. [Algoritmik Detaylar](https://www.google.com/search?q=%23algoritmik-detaylar)
10. [Lisans](https://www.google.com/search?q=%23lisans)

## Sistem Mimarisi

VibeLens işlem hattı (pipeline) dört farklı aşamadan oluşur:

1. **Görsel Alım & Analiz:**
* API, `multipart/form-data` isteği aracılığıyla bir görüntü dosyası kabul eder.
* **Yüz Algılama:** Yüzleri tespit etmek ve demografik verileri (yaş, cinsiyet) çıkarmak için `DeepFace` (RetinaFace arka ucu) kullanır.
* **Duygu Tanıma:** Yüz bölgesini kırpar ve 8 farklı duygu için ham logit değerleri üretmek üzere `HSEmotion`'a (ENet tabanlı bir PyTorch modeli) iletir.


2. **Dinamik Duygu Puanlama:**
* Ham logitler, önceden tanımlanmış eşiklere göre "Göreceli Gücü" hesaplayan özel bir algoritmadan geçirilir.
* Sistem, karmaşık bir duygusal profil oluşturmak için bir "Baskın Duygu" ve ince bir "İkincil Duygu" tanımlar (örn. "Hafif Öfke tonlu Üzüntü").


3. **Üretken Kürasyon (LLM):**
* Duygusal profil, demografik verilerle birlikte yapılandırılmış bir istem (prompt) haline getirilir.
* İçerik önerileri oluşturmak için **Google Gemini Flash** sorgulanır. İstem mühendisliği (prompt engineering), genel sonuçlardan kaçınmak için çeşitlilik kurallarını uygular (örn. "Gişe Rekortmenleri" yerine "Gizli Cevherler").


4. **Metadata Zenginleştirme:**
* LLM'den dönen ham başlıklar, harici API'ler (TMDB, iTunes, Google Books) kullanılarak metadata (Posterler, Puanlar, Özetler, Yıllar) ile zenginleştirilir.
* Bu süreç, gecikmeyi en aza indirmek için birden fazla öğe verisini paralel olarak getirmek adına `concurrent.futures` kullanır.



## Temel Özellikler

* **Yüksek Performanslı API:** Asenkron istek yönetimi ve otomatik OpenAPI dokümantasyonu sunan FastAPI üzerine inşa edilmiştir.
* **Gelişmiş Bilgisayarlı Görü:** Yüz ifadesi tanıma (FER) için son teknoloji modelleri yüksek doğrulukla entegre eder.
* **Karmaşık Duygu Mantığı:** İkincil duyguları ve ağırlıklı puanları hesaplayarak basit "Mutlu/Üzgün" sınıflandırmasının ötesine geçer.
* **Yapılandırılmış YZ Çıktısı:** Güvenilir ayrıştırma ve tip güvenliği sağlamak için LLM'den gelen çıktıları JSON şemasına zorlar.
* **Güçlü Metadata Toplama:** Birden fazla sağlayıcıyı (TMDB, iTunes, Open Library) sorgulayan ve resmi API'ler başarısız olursa DuckDuckGo kazımasına (scraping) geçen dayanıklı bir geri dönüş (fallback) mekanizması.
* **Eşzamanlılık (Concurrency):** Ağır I/O işlemleri (harici API çağrıları), yanıt süresinin düşük kalmasını sağlamak için iş parçacıklarına (threaded) ayrılmıştır.

## Teknoloji Yığını

* **Dil:** Python 3.11+
* **Web Çatısı:** FastAPI / Uvicorn
* **Bilgisayarlı Görü:**
* PyTorch
* DeepFace
* HSEmotion (HSE-as/hsemotion)
* OpenCV (cv2)


* **Üretken YZ:** Google Generative AI (Gemini Flash)
* **Veri Doğrulama:** Pydantic
* **HTTP İstemcisi:** Requests
* **Arama/Kazıma:** DuckDuckGo Search
* **Süreç Yönetimi:** Concurrent Futures (ThreadPoolExecutor)

## Ön Koşullar

* Python 3.10 veya üzeri.
* pip (Python Paket Yükleyicisi).
* Geçerli bir Google Gemini API Anahtarı.
* Geçerli bir TMDB (The Movie Database) API Anahtarı.

## Kurulum

1. **Depoyu (Repository) Klonlayın**
```bash
git clone https://github.com/kullaniciadiniz/vibelens-backend.git
cd vibelens-backend

```


2. **Sanal Ortam (Virtual Environment) Oluşturun**
Bağımlılıkları yönetmek için sanal ortam kullanılması önerilir.
```bash
python -m venv .venv
# Windows'ta etkinleştirme:
.venv\Scripts\activate
# macOS/Linux'ta etkinleştirme:
source .venv/bin/activate

```


3. **Bağımlılıkları Yükleyin**
```bash
pip install -r requirements.txt

```



## Yapılandırma

Uygulama, API kimlik doğrulaması için ortam değişkenlerine ihtiyaç duyar. Kök dizinde `.env.example` dosyasını baz alarak bir `.env` dosyası oluşturun.

**Gerekli Değişkenler:**

| Değişken | Açıklama |
| --- | --- |
| `GEMINI_API_KEY` | Google Gemini için API Anahtarı (YZ üretimi). |
| `TMDB_API_KEY` | The Movie Database için API Anahtarı (Film/Dizi metadatası). |

**PyTorch Güvenliği Üzerine Not:**
Proje, HSEmotion kütüphanesi tarafından kullanılan eski model ağırlıklarını desteklemek için `torch.load` yaması (patch) içerir. Bu işlem `app/core/models.py` içinde dahili olarak yönetilir.

## Kullanım

### Sunucuyu Çalıştırma

Uygulamayı Uvicorn kullanarak başlatın. Sunucu `http://127.0.0.1:8000` adresinde çalışacaktır.

```bash
uvicorn main:app --reload

```

### API Uç Noktaları (Endpoints)

* **GET /**: Servis sağlığını gösteren HTML durum sayfasını sunar.
* **POST /analyze**: Ana analiz uç noktası.
* **Form Verisi:**
* `file`: Analiz edilecek görüntü dosyası (JPEG/PNG).
* `category`: İstenen öneri kategorisi (`Movie`, `Series`, `Book`, `Music`).


* **Yanıt:** Algılanan ruh halini, demografik bilgileri ve öneri listesini içeren JSON nesnesi.



### Canlı Kamera Testi

Bilgisayarlı Görü mantığını ve duygu eşiklerini web kameranızı kullanarak gerçek zamanlı test etmek için bağımsız (standalone) bir betik sağlanmıştır.

```bash
python live_camera_emotion_test.py

```

## Proje Yapısı

Proje; yapılandırma, şemalar, servisler ve API yönlendirmeleri arasındaki endişeleri (concerns) ayırmak için modüler bir mimari izler.

```text
VibeLensBackend/
├── app/
│   ├── api/
│   │   └── router.py           # API rota tanımları ve istek yönetimi
│   ├── core/
│   │   ├── config.py           # Ortam değişkeni yönetimi
│   │   ├── models.py           # ML model başlatma ve genel sabitler
│   │   └── prompts.py          # LLM istem mühendisliği mantığı
│   ├── schemas/
│   │   └── analysis.py         # Pydantic modelleri ve Enum'lar
│   ├── services/
│   │   ├── llm_services.py     # Google Gemini ile etkileşim
│   │   ├── search_service.py   # Harici API entegrasyonu (TMDB, iTunes vb.)
│   │   └── vision_service.py   # Görüntü işleme ve duygu tanıma mantığı
│   └── utils/
│       └── timer.py            # Performans izleme için zamanlama aracı
├── static/
│   ├──  index.html             # Statik durum sayfası
├── .env.example                # Ortam değişkenleri için şablon
├── .gitignore                  # Git hariç tutma kuralları
├── live_camera_emotion_test.py # Bağımsız CV test betiği
├── main.py                     # Uygulama giriş noktası
├── README.md                   # Proje dokümantasyonu
└── requirements.txt            # Python bağımlılıkları

```

## Algoritmik Detaylar

### Dinamik Duygu Puanlama

Standart duygu tanıma modelleri genellikle "Nötr"ü destekleyen ham olasılıklar verir veya ince ifadeleri yakalamakta başarısız olur. VibeLens, `app/services/vision_service.py` içinde özel bir algoritma uygular:

1. **Eşikleme (Thresholding):** Her duygunun (örn. Korku, Mutluluk) belirli bir hassasiyet eşiği vardır.
2. **Ağırlıklı Güç:** Puan, `Ham Skor / Eşik` olarak hesaplanır. Bu, doğal olarak daha düşük olasılıklara sahip duyguların (Korku gibi), baskın duygularla (Mutluluk gibi) rekabet etmesini sağlar.
3. **İkincil Duygu Tespiti:** Sistem, nüans sağlamak için en yüksek ağırlığa sahip ikinci duyguyu tanımlar (örn. bir yüzü sadece "Üzgün" olarak değil, "Küçümseme içeren Üzüntü" olarak sınıflandırmak).
4. **Normalizasyon:** Nihai puanlar, LLM bağlamı için gerekli verileri korurken kullanıcı arayüzü için baskın duyguyu vurgulayacak şekilde normalleştirilir.

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için LICENSE dosyasına bakın.
