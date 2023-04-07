import socket
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
def Main():
    host='127.0.0.1'
    port=12341
    s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    s.connect((host,port))
    print("Hello, I’m the Oracle. How can I help you today?")
    while True:
        message = input("\nInput the message: ")
        s.send(message.encode('ascii'))
        data=s.recv(1024)
        data=data.decode('ascii')
        if '_' in data:
            types=message.split(':')[0]
            dates=message.split(':')[1]
            long=float(data.split("_")[0])
            lat=float(data.split("_")[1])
            print('Received from the server: The location of %s in %s is (%.2f, %.2f).' %(types,dates,long,lat),'Generating map for you.')
            plt.figure()
            mymap = Basemap(llcrnrlon=long-20, llcrnrlat=lat-20, urcrnrlon=long+20, urcrnrlat=lat+20, resolution='h', projection='merc')
            mymap.drawcoastlines()
            mymap.fillcontinents()
            mymap.drawparallels(np.arange(round((lat-20)/10)*10, round((lat+20)/10)*10, 5), labels=[1, 1, 0, 1])
            mymap.drawmeridians(np.arange(round((long-20)/10)*10, round((long+20)/10)*10, 10), labels=[1, 1, 0, 1])
            xpt, ypt = mymap(long, lat)
            mymap.plot(xpt, ypt, 'c*', markersize=12, c='r')
            xptt, yptt = mymap(long + 0.5, lat + 0.5)
            location_ = "$(" + "%.2f"%long + "'^\circ," + "%.2f"%lat + "^\circ)$"
            plt.text(xptt, yptt, location_, c='r', fontsize=8, style="italic", weight="light", )
            plt.show()
        else:
            print('Received from the server: %s' %data)
        ans=input('\nDo you want to continue(y/n) :')
        if ans=='y':
            continue
        else:
            print('Bye')
            break
    s.close()
if __name__=='__main__':
    Main()

