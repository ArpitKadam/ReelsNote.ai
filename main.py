from src.helpers.reel_scrapper import Scraper
import json

scraper = Scraper()
output = scraper.download("https://www.instagram.com/reel/DasfzZBxRTg/?utm_source=ig_web_copy_link&igsh=MzRlODBiNWFlZA==")
print(json.dumps(output, indent=2, ensure_ascii=False))