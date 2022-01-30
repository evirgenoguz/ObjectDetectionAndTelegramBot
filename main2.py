import cv2
import requests


#telegram mesaj gönderme fonksiyonumuz http protokolu ile post isteği yapıyoruz
def SendTelegramMessage(object_name, status):
    
    if(status == 0):
        mesaj = f"{object_name} nesnesi kamera açısından çıktı"
        x = requests.post(url = "https://api.telegram.org/bot5217908223:AAGVbJMvMHbIQlzSk-9rfcw7NRJalPFYRyw/sendMessage", data = {"chat_id":"840439204", "text":mesaj})
        print(x.text)
        print(f"{object_name} nesnesi kamera açısından çıktı")
    elif(status == 1):
        pass
        #print(f"{object_name} nesnesi kamera açısına girdi")

#nesne tespiti için kullanacağımız threshold eşiği
thresh = 0.4
#Opencv DNN
#hazır model kullandık
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size = (320, 320), scale = 1/255)
model.setInputSize(320,320)
model.setInputScale(1.0/ 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


#laptop kamerasını açtık
#sunum için hazır video kullandık
cap = cv2.VideoCapture("VID-20220128-WA0005.mp4")



#sınıfların listesini txt dosyasından alma
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        #print(class_name)
        class_name = class_name.strip()
        classes.append(class_name)
        
#classes.txt dosyasındaki nesneleri konsola yazdırmak        
print("nesneler")
print(classes)        

#bu listemiz kamera açısında belli bir süre duran nesnelerin indisindeki
#false değerini true ya çevirerek bu nesnenin kamera açısında olduğunu kesinleştirir
fixture_classes = [False] * len(classes)
print(fixture_classes)



#listemiz 30 framedan fazla duran nesneyi demirbaş olarak ayarlamak için oluşturulmuştur
classes_exist = [0] * len(classes)
print(classes_exist)





#kamerayı döngüye aldık
while True:
    
    #frameleri almak için
    ret, frame = cap.read()

    #nesne tespiti burada yapılır ve değişkenlere sonuçlar atanır.
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold = thresh)
    
   
    #bu listemiz ise indisine göre hangi nesnenin kamera açısında oldugunu gösterir
    my_list = [0] * len(classes)
   
    #frame üzerinde çizim yapmak için kullanılan for döngüsü
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        #nesnenin hangi kordinatlarda olduğunu anlamamız için
        x, y, w, h = bbox
        #print(x, y, w, h)
        
        #class_id ye göre class ın ismini almamız için
        class_name = classes[class_id]
        
        
        #burada ise kare çizmek ve class ın ismini yazmak
        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)
        
        
        my_list[class_id] += 1
        classes_exist[class_id] += 1
        
        
            
       
        #class 30 framedan fazla kamera açısındaysa demirbaş olur ve telegram mesajı 1 kodu ile gönderilir.
        if(classes_exist[class_id] >= 30):
            classes_exist[class_id] = 30
            fixture_classes[class_id] = True
            SendTelegramMessage(classes[class_id], 1)
            
            
        
    #eğer kamera açısında nesne yoksa 30 olan sayı tek tek azalır        
    if(my_list[class_id] == 0):
        classes_exist[class_id] -= 1
        
        
    #eğer daha önce nesne kamera açısındaysa demirbaş listesinde true olmuştur
    #30 sayısı eksilerek 0 olduysa nesne kamera açısından çıkmıştır demek ve 
    #bu durumda telegram mesajı 0 kodu ile gönderilir    
    if(classes_exist[class_id] <= 0 and fixture_classes[class_id] == True):
        fixture_classes[class_id] = False
        classes_exist[class_id] = 0
        SendTelegramMessage(classes[class_id], 0)
        print("******************************")
    
    #kontrol için yazılmıştır
    #print(classes_exist)
    #print(fixture_classes)
    #print(my_list)
    
    #kontrol için yazılmıştır
    #print("class ids: ", class_ids)
    #print("score: ", scores)
    #print("bbox: ", bboxes)
    
    
    #kamerayı kapatmak için q tuşu atandı
    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break