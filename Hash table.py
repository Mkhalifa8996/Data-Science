# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 09:20:10 2025

@author: Mkhal
"""
import csv
class HashTable:
    def __init__(self):
        self.MAX = 10
        self.arr = [[] for i in range(self.MAX)]

    def getHash(self, key):
        h= 0
        for ch in key:
            h += ord(ch)
        return h % self.MAX
    
    # Collision is handled here using chaining (linked lists)
    def add(self, key, value):
        h = self.getHash(key)
        
        found = False
        for i, val in enumerate(self.arr[h]):
            if len(val) == 2 and val[0] == key:
                self.arr[h][i] = (key, value)
                found = True
                break
        if not found:
            self.arr[h].append((key, value))
    def addLinearPrbing(self, key, value):
        h = self.getHash(key)
        
        
        if len(self.arr[h]) == 0 or self.arr[h][0] == key:
            self.arr[h] = (key, value)
            return
        
        if len(self.arr[h]) == 2:
            for i, val in enumerate(self.arr):
                if len(val) == 0:
                    self.arr[i] = (key, value)
                    break
        
    
    def get(self, key):
        h = self.getHash(key)
        if not isinstance(self.arr[h], list):
            return self.arr[h][1]
        
        for element in self.arr[h]:
            if element[0] == key:
                return element[1]
    
    def delete(self, key):
        h = self.getHash(key)
        
        for idx, val in enumerate(self.arr[h]):
            if self.arr[h][idx] == key:
                del self.arr[h][idx]
                
    def averageTemperature(self, days):
        avg = 0
        key = ''
        for i, val in enumerate(days):
            key = 'Jan ' + (str(val))
            avg += int(self.get(key))
        return avg/len(days)
        
if __name__ == '__main__':
    
    ht = HashTable()
    
    with open('nyc_weather.csv', 'r') as f:
        reader = csv.reader(f, delimiter = ',')
        # Ignore header row
        next(reader)
    
        for line in reader:
            key = line[0]
            value = line[1]
            
            # Insert using chaining (list)
            # ht.add(key, value)  
            
            # Insert using linear probing 
            ht.addLinearPrbing(key, value)
        
    print(ht.arr)

    # Average temperatures for one week
    d = [str(i) for i in range(1,8)]
    print('Average temperatures for one week :',ht.averageTemperature(d))
        
    # Average temperatures for 10 days
    d = [str(i) for i in range(1,11)]
    print('Average temperatures for 10 days : ' , ht.averageTemperature(d))
    
    # temperature on Jan 9
    print('temperature on Jan 9 : ', ht.get('Jan 9'))

    # temperature on Jan 4
    print('temperature on Jan 9 : ', ht.get('Jan 4'))
