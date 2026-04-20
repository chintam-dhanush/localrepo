import java.security.MessageDigest;         
import java.security.NoSuchAlgorithmException;
import java.util.Scanner;
 public class cns7a {
    public static String generateMD5(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5"); 
            byte[] hashBytes = md.digest(input.getBytes()); 
            
            StringBuilder hexString = new StringBuilder();
            for (byte b : hashBytes) {
                hexString.append(String.format("%02x", b));
            }
            return hexString.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("MD5 algorithm not found", e);
        }
    }
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter input1 string for MD5: ");
        String input1 = sc.nextLine();
        
        System.out.println("MD5 Hash for input1: " + generateMD5(input1));
        System.out.print("Enter input2 with small change in the string : ");
        String input2 = sc.nextLine();
        
        System.out.println("MD5 Hash for input2: " + generateMD5(input2));

        sc.close();
    }
}