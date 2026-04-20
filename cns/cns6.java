import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;
public class cns6{
    public static void main(String[] args){
        String msg="this is secret";
        String secretkey="123";
        try{
            byte[] hmac=generateMac(msg,secretkey);
            String hmacBase64=Base64.getEncoder().encodeToString(hmac);
            System.out.println("HMAC: "+hmacBase64);
        }catch(NoSuchAlgorithmException | InvalidKeyException e){
            e.printStackTrace();
        }
    }
    public static byte[] generateMac(String msg,String secretKey) throws NoSuchAlgorithmException,InvalidKeyException{
        Mac mac=Mac.getInstance("HmacSHA256");
        SecretKeySpec secretKeySpec=new SecretKeySpec(secretKey.getBytes(),"HmacSHA256");
        mac.init(secretKeySpec);
        return mac.doFinal(msg.getBytes());
    }
}