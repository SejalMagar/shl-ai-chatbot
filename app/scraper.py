from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options

from webdriver_manager.chrome import ChromeDriverManager

import json
import time

options = Options()

options.add_argument("--start-maximized")

driver = webdriver.Chrome(
    service=Service(
        ChromeDriverManager().install()
    ),
    options=options
)

url = "https://www.shl.com/solutions/products/product-catalog/"

driver.get(url)

time.sleep(15)

links = driver.find_elements(By.TAG_NAME, "a")

catalog = []

seen = set()

for link in links:

    try:

        href = link.get_attribute("href")

        text = link.text.strip()

        if href and "/products/" in href and text:

            if href not in seen:

                seen.add(href)

                catalog.append({
                    "name": text,
                    "url": href,
                    "description": (
                        f"{text} assessment for "
                        "professional hiring evaluation"
                    ),
                    "test_type": "Assessment",
                    "skills": [
                        text.lower()
                    ]
                })

    except:
        pass

driver.quit()

with open("data/shl_catalog.json", "w") as f:

    json.dump(catalog, f, indent=2)

print(f"Saved {len(catalog)} assessments.")