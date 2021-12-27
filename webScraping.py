import requests
from bs4 import BeautifulSoup


def Web_scraping(root):

    website = f'{root}/movies'
    result = requests.get(website)
    content = result.text
    soup = BeautifulSoup(content, 'lxml')

    box = soup.find('article', class_='main-article')

    links = [link['href'] for link in box.find_all('a', href=True)]

    for link in links:
        result = requests.get(f'{root}/{link}')
        content = result.text
        soup = BeautifulSoup(content, 'lxml')

        box = soup.find('article', class_='main-article')
        title = box.find('h1').get_text()
        transcript = box.find(
            'div', class_='full-script').get_text(strip=True, separator=' ')
        i = 0
        while i < 10:
            with open('Script.txt', 'a', encoding="utf-8") as file:
                file.write(transcript)
            i = i + 1


Web_scraping('https://subslikescript.com')
