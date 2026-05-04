import java.math.BigInteger;
import java.security.SecureRandom;
import java.util.Scanner;

public class cns5 {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        SecureRandom random = new SecureRandom();
        
        System.out.print("Enter a prime number (p): ");
        BigInteger p = scanner.nextBigInteger();
        System.out.print("Enter a primitive root modulo p (g): ");
        BigInteger g = scanner.nextBigInteger();
  
        BigInteger a = new BigInteger(1024, random);
        BigInteger A = g.modPow(a, p);
     
        BigInteger b = new BigInteger(1024, random);
        BigInteger B = g.modPow(b, p);
  
        System.out.println("Alice's Public Key (A): " + A);
        System.out.println("Bob's Public Key (B): " + B);
    
        BigInteger secretKeyAlice = B.modPow(a, p);
        BigInteger secretKeyBob = A.modPow(b, p);
   
        System.out.println("Alice's Secret Key: " + secretKeyAlice);
        System.out.println("Bob's Secret Key: " + secretKeyBob);
        if (secretKeyAlice.equals(secretKeyBob)) {
            System.out.println("Key exchange successful. Shared secret key is established.");
        } else {
            System.out.println("Key exchange failed.");
        }
        scanner.close();
    }
}