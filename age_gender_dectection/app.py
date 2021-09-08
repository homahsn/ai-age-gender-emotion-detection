# ~~~~~~~~~~ MODULES ~~~~~~~~~~ # 
import cv2
import requests
import pandas as pd
from os import system
from os import listdir
from time import sleep
from os.path import isfile
from getpass import getpass
from instaloader import Post
from instaloader import Instaloader


from comp.plot_results import compare as evaluate
from comp.smahesh import detect as smahesh
from comp.agender import detect as agender
from comp.conf import export_file_name
from comp.conf import labels_file_name


# ~~~~~~~~~~ CONSTANTS ~~~~~~~~~~ #
username = ''
password = ''


# ~~~~~~~~~~ LOGIN ~~~~~~~~~~ #
def login():
    insta = Instaloader()
    try:
        insta.load_session_from_file(username, 'comp/session.txt')
        return insta
    except:
        try:
            insta.login(
                input('Enter your Instagram\'s username: '),
                getpass('Enter your Instagram\'s password: ')
            )
            insta.save_session_to_file('comp/session.txt')
            return insta
        except:
            print('Unable to login!')


# ~~~~~~~~~~ DOWNLOAD ~~~~~~~~~~ #
def download():
    insta = login()
    if not insta == None:
        try:
            with open('shortcodes.txt') as f:
                codes = f.read().split('\n')
        except:
            pass
        images = [i.split('.')[0] for i in listdir('cache')]
        codes = [i for i in codes if i not in images]
        for c in codes:
            try:
                print(f'{codes.index(c)+1}/{len(codes)} Downloading image {c} ...', end=' ')
                sleep(1)
                url = Post.from_shortcode(insta.context, c).url
                response = requests.get(url)
                with open(f'cache/{c}.jpg', 'wb') as f:
                    f.write(response.content)
                print()
            except:
                print('FAILED!')
    else:
        print('Unable to login!')


# ~~~~~~~~~~ MAIN ~~~~~~~~~~ #
def run():
    images = [i for i in listdir('cache') if i.endswith('.jpg')] 
    for i in images:
        print('-'*50)
        try:
            t = f'image {images.index(i)+1}/{len(images)}'
            print(t, f'[{i}]'.ljust(15))
            r1 = smahesh(cv2.VideoCapture(f'cache/{i}'))
            r2 = agender(f'cache/{i}')
            r1 = '\tsmahesh:     ' + ' | '.join([f'{i[0]}, {i[1]}' for i in r1])
            r2 = '\tagender:     ' + ' | '.join([f'{i[0]}, {i[1]}' for i in r2])
            print(r1, r2, sep='\n')
        except:
            pass


# ~~~~~~~~~~ HELP ~~~~~~~~~~ #
def help():
    print(f'''
    h, help         Help
    i, input        Input a shortcode
    d, download     Download all new images (Based on shotrcodes.txt)
    r, run          Run tests on all the downloaded images
    x, export       Run tests and export result as a csv file
    v, evaluate     Evaluate results with "{labels_file_name}" file
    q, quit         Exit
    ''')

# ~~~~~~~~~~ MENU ~~~~~~~~~~ #
def menu():
    while True:
        dict(
            h=help,     help=help,
            i=single,   input=single,
            d=download, download=download,
            r=run,      run=run,
            x=export,   export=export,
            v=evaluate, evaluate=evaluate,
            q=exit,     quit=exit,
        ).get(input('(h : help) >> ').lower(), lambda:print('Enter "h" to see help.'))()


# ~~~~~~~~~~ SINGLE ~~~~~~~~~~ #
def single():
    insta = login()
    if not insta == None:
        c = input('Enter the shotrcode: ')
        try:
            print('Downloading image ...', end=' ')
            url = Post.from_shortcode(insta.context, c).url
            response = requests.get(url)
            with open(f'cache/{c}.jpg', 'wb') as f:
                f.write(response.content)
            print()
        except:
            print('FAILED!')
        try:
            r1 = smahesh(cv2.VideoCapture(f'cache/{c}.jpg'))
            r2 = agender(f'cache/{c}.jpg')
            r1 = '\tsmahesh:     ' + ' | '.join([f'{i[0]}, {i[1]}' for i in r1])
            r2 = '\tagender:     ' + ' | '.join([f'{i[0]}, {i[1]}' for i in r2])
            print(r1, r2, sep='\n')
        except:
            pass
    else:
        print('Unable to login!')


# ~~~~~~~~~~ EXPORT ~~~~~~~~~~ #
def export():
    # ~~~~~~~~~~~~~~~~~~~~ #
    def cleanify(p):
        if not p[1]:
            return None, None
        g = p[0].strip().lower()
        a = p[1]
        if '-' in str(a):
            a = sum([int(i) for i in a.split('-')]) // 2
        return g, a
    # ~~~~~~~~~~~~~~~~~~~~ #
    images = [i for i in listdir('cache') if i.endswith('.jpg')]
    res = []
    for i in images:
        c = i.split('.')[0]  # remove ".jpg"
        try:
            print(f'image {images.index(i)+1}/{len(images)}', end='\r')
            r = smahesh(cv2.VideoCapture(f'cache/{i}'))
            for j in r:
                g, a = cleanify(j)
                res.append(dict(code=c, network='smahesh', gender=g, age=a))
            
            r = agender(f'cache/{i}')
            for j in r:
                g, a = cleanify(j)
                res.append(dict(code=c, network='agender', gender=g, age=a))
        except:
            pass
    pd.DataFrame(res).to_csv(export_file_name, index=False)
    print(export_file_name, 'has been saved.')


# ~~~~~~~~~~ ~~~~~~~~~~ ~~~~~~~~~~ #
if __name__ == '__main__':
    system('cls')
    system('clear')
    menu()
