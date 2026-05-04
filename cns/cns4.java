import java.security.*;
import javax.crypto.Cipher;
import java.util.Base64;
import java.util.Scanner;

public class cns4 {

    private static KeyPair genKeys() throws Exception {
        KeyPairGenerator keyGen = KeyPairGenerator.getInstance("RSA");
        keyGen.initialize(512); 
        return keyGen.generateKeyPair(); 
    }


    private static String enc(String pt, PublicKey pubKey) throws Exception {
        Cipher cipher = Cipher.getInstance("RSA"); 
        cipher.init(Cipher.ENCRYPT_MODE, pubKey); 
        byte[] encBytes = cipher.doFinal(pt.getBytes());
        return Base64.getEncoder().encodeToString(encBytes); 
    }

   
    private static String dec(String et, PrivateKey privKey) throws Exception {
        Cipher cipher = Cipher.getInstance("RSA"); 
        cipher.init(Cipher.DECRYPT_MODE, privKey); 
        byte[] decBytes = Base64.getDecoder().decode(et); 
        byte[] ptBytes = cipher.doFinal(decBytes); 
        return new String(ptBytes); 
    }

    public static void main(String[] args) {
        try (Scanner scanner = new Scanner(System.in)) {
           
            KeyPair keyPair = genKeys();
            PublicKey pubKey = keyPair.getPublic();
            PrivateKey privKey = keyPair.getPrivate();

           
            System.out.println("Public Key: " + Base64.getEncoder().encodeToString(pubKey.getEncoded()));
            System.out.println("Private Key: " + Base64.getEncoder().encodeToString(privKey.getEncoded()));

           
            System.out.print("Enter plaintext message: ");
            String pt = scanner.nextLine();

           
            String et = enc(pt, pubKey);
            System.out.println("Encrypted: " + et);

          
            String dt = dec(et, privKey);
            System.out.println("Decrypted: " + dt);

        } catch (Exception e) {
            e.printStackTrace(); 
        }
    }
}