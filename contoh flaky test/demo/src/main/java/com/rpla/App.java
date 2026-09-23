package com.rpla;

import java.util.Random;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        System.out.println( "Hello World!" );
    }

    /**
     * Method ini mengembalikan nilai secara acak.
     * Penggunaan random (atau nilai yang bergantung pada state eksternal) 
     * sering menjadi penyebab terjadinya Flaky Test.
     */
    public int generateRandomNumber() {
        Random random = new Random();
        // Mengembalikan angka acak dari 0 hingga 9
        return random.nextInt(10);
    }
}
