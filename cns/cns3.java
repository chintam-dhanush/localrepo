
import javax.crypto.Cipher;
import javax.crypto.KeyGenerator;
import javax.crypto.SecretKey;
// import javax.crypto.spec.SecretKeySpec;
import java.util.Base64;

public class cns3 {


    private static SecretKey genKey() throws Exception {
        KeyGenerator kg = KeyGenerator.getInstance("DES");
        kg.init(56); // DES uses a 56-bit key
        return kg.generateKey();
    }


    private static String enc(String pt, SecretKey k) throws Exception {
        Cipher c = Cipher.getInstance("DES");
        c.init(Cipher.ENCRYPT_MODE, k);
        byte[] encBytes = c.doFinal(pt.getBytes());
        return Base64.getEncoder().encodeToString(encBytes);
    }

    private static String dec(String et, SecretKey k) throws Exception {
        Cipher c = Cipher.getInstance("DES");
        c.init(Cipher.DECRYPT_MODE, k);
        byte[] decBytes = Base64.getDecoder().decode(et);
        byte[] ptBytes = c.doFinal(decBytes);
        return new String(ptBytes);
    }

    public static void main(String[] args) {
        try {
            // Generate DES key
            SecretKey k = genKey();

            // Print key in Base64 format
            System.out.println("Key: " + Base64.getEncoder().encodeToString(k.getEncoded()));

            // Sample plaintext
            String pt = "Hello, World!";
            System.out.println("Plaintext: " + pt);

            // Encrypt plaintext
            String et = enc(pt, k);
            System.out.println("Encrypted: " + et);

            // Decrypt ciphertext
            String dt = dec(et, k);
            System.out.println("Decrypted: " + dt);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}