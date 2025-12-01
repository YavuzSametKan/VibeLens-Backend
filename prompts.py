import json
from models import Category


def build_gemini_prompt(category: Category, age: int, gender: str, emotion: str, raw_scores: dict) -> str:
    # Ham skorları string'e çevir
    scores_str = json.dumps(raw_scores)

    # Ortak Giriş
    base_prompt = f"""
    Sen VibeLens, gelişmiş bir duygu analisti ve kültür asistanısın.

    KULLANICI PROFİLİ:
    - Yaş: {age}, Cinsiyet: {gender}
    - Baskın Duygu: {emotion}
    - Detaylı Duygu Dağılımı: {scores_str}

    GÖREV:
    Kullanıcının duygu dağılımını (sadece baskın olana bakma, alt metinleri de oku) yorumla ve ona en uygun '{category.value}' önerilerini yap.
    """

    # Kategoriye Özel Talimatlar
    if category == Category.MOVIE:
        instruction = """
        - Önerilerinde 'creator' kısmına Yönetmen adını yaz.
        - Hem popüler hem de sanatsal değeri olan filmler seç.
        - 'poster_url' için orijinal film afişi görseli linki ver.
        """
    elif category == Category.SERIES:
        instruction = """
        - 'creator' kısmına Başrol Oyuncularını veya Yapımcıyı yaz.
        - Bölüm süresi ve türüne göre (Örn: Mini Dizi, Sitcom) seçim yap.
        """
    elif category == Category.MUSIC:
        instruction = """
        - Tek şarkı veya Albüm önerebilirsin.
        - 'creator' kısmına Sanatçı/Grup adını yaz.
        - Modu yükseltecek veya eşlik edecek parçalar seç.
        """
    elif category == Category.BOOK:
        instruction = """
        - 'creator' kısmına Yazar adını yaz.
        - Yaş grubuna uygun edebi derinlikte kitaplar seç.
        """

    # Çıktı Formatı (Asla Değişmez)
    output_format = """
    ⚠️ TEKNİK FORMAT KURALLARI (Çok Önemli):
    1. Cevabı SADECE geçerli bir JSON objesi olarak ver.
    2. 'mood_description' alanını maksimum 2 cümle ile tut. KISA VE ÖZ OL.
    3. 'reason' alanlarını tek bir çarpıcı cümle ile açıkla. UZATMA.
    ÇIKTI FORMATI (SADECE JSON):
    {
        "mood_title": "Yapay Zeka Yorum Başlığı",
        "mood_description": "Duygu analizine dayalı samimi açıklama.",
        "recommendations": [
            {
                "title": "Eser Adı",
                "creator": "Yaratıcı",
                "rating": "Puan",
                "poster_url": "Link",
                "reason": "Neden?"
            }
        ]
    }
    """

    return base_prompt + instruction + output_format