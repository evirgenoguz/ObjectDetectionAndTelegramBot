import cv2
import requests



#bu fonksiyonumuz gelen status koduna göre girdi veya çıktı mesajını telegramdan gönderir
def SendTelegramMessage(object_name, status):    
    if(status == 0):
        mesaj = f"{object_name} nesnesi kamera açısından çıktı"
        x = requests.post(url = "https://api.telegram.org/bot5217908223:AAGVbJMvMHbIQlzSk-9rfcw7NRJalPFYRyw/sendMessage", data = {"chat_id":"840439204", "text":mesaj})
        print(x.text)
        print(f"{object_name} nesnesi kamera açısından çıktı")
    elif(status == 1):     
        print(f"{object_name} nesnesi kamera açısına girdi")



#threshold değerimiz
thresh = 0.4

#Opencv DNN
#hazır yolov4 modelini kullandık
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)

#model için bazı parametreleri değiştirdik algılaması kolay olması için
model.setInputParams(size = (320, 320), scale = 1/255)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


#normalde kamerada çalışıyor fakat sunum için bir video ekledik
cap = cv2.VideoCapture("VID-20220128-WA0005.mp4")



#sınıfların listesini txt dosyasından alma
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        #print(class_name)
        class_name = class_name.strip()
        classes.append(class_name)
        
#nesneleri konsola yazdırdık
print("nesneler")
print(classes)        

#nesneler eğer 30 frameden fazla gözükürse burası true olur ama 1 den fazla aynı nesneden olursa nasıl bir yol izleriz belli değil şimdilik
fixture_classes = [False] * len(classes)
print(fixture_classes)

#bu liste 0 larla dolu bu listeyi kullanma amacım 
#eğer kamera açısına nesne girerse o nesnenin classes sınıfındaki hangi indexte oldugunu algılamak
#için eğer kamera açısına nesne girerse o indexteki sayı artıcak
classes_exist = [0] * len(classes)
print(classes_exist)





#kamerayı döngüsü
while True:
    
    #frameleri almak için
    ret, frame = cap.read()

    #nesne tespiti yapmak için kullanılan kod satırı ve verilerinin atandığı değişkenler
    (class_ids, scores, bboxes) = model.detect(frame, confThreshold = thresh)
    
    
    #bu listede oluşturma nedenim artan indise göre kamerada hangi nesnenin olduğunu anlayacağız
    #ve bu listedeki sayı classes_exist listesindeki aynı indisteki sayıyı artırır ve bu artan sayı 30 u geçerse 
    #nesnemiz fixture_classes listesindeki false yi Trueya çevirir
    #bu da bize nesnenin kesin olarak kamera açısında oldugunu gösterir
    my_list = [0] * len(classes)
   
    
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        
        x, y, w, h = bbox
        #print(x, y, w, h)
        
        class_name = classes[class_id]
        
        #opencv ile nesneyi kareya alma ve class ismini yazma işlemi
        cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (200, 0, 50), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 0, 50), 3)
        
        #while döngüsü içinde tanımlandığı için her while tekrarlandığında 0 lanır böylelikle hangi nesnelerden frame içinde
        #kaçar tane olduğunu tespit edebiliriz
        my_list[class_id] += 1
        
        #bu liste while dışında tanımlandığı için my_list içeriği gibi fakat sürekli artacaktır
        #aşağıda bu artım sayısını en fazla 30 olacak şekilde ayarladım
        classes_exist[class_id] += 1
        
        
            
        
        #artan sayı en fazla 30 olacak şekilde ayarladım ve eğer bu sayı 30 ollursa 
        #nesnenin kamera açısında oldugu kesinleştirilir ve fixture_classes listesindeki
        #o indisteki eleman True olarak değiştirilir.
        if(classes_exist[class_id] >= 30):
            classes_exist[class_id] = 30
            fixture_classes[class_id] = True
            SendTelegramMessage(classes[class_id], 1)
            
            
        
    #eğer nesne kamera açısından çıkarsa my_list e o indis 0 olacağı için
    #nesne açıdan çıktıgında classes_exist teki 30 sayısı teker teker düşecektir 
    if(my_list[class_id] == 0):
        classes_exist[class_id] -= 1
        
        
    #classes_exis listesinde yukarıdaki if içine girip düşen sayı 0 a ulaştıgında 
    #bu nesnenin kamera açısından çıktığını algılıyoruz ve daha önce kamera açısında olup olmadıgını kontrol
    #etmek için fixture_classes içindeki değeri kontrol ediyoruz True ise daha önce nesne kamera açısındadır demektir
    if(classes_exist[class_id] <= 0 and fixture_classes[class_id] == True):
        fixture_classes[class_id] = False
        classes_exist[class_id] = 0
        SendTelegramMessage(classes[class_id], 0)
        
    
    #alt taraftaki yorum satırlarını açarak kontrol sağlayabilirsiniz
     
    #print(classes_exist)
    #print(fixture_classes)
    #print(my_list)
    
  
    
    
    #kamerayı q tuşu ile kapatmak için
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    
    
    
