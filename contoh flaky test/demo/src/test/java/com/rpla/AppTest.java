package com.rpla;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Unit test for simple App.
 */
public class AppTest 
{
    /**
     * Contoh Flaky Test 1: Bergantung pada nilai Random
     */
    @Test
    void testRandomNumberIsFive()
    {
        App app = new App();
        int result = app.generateRandomNumber();
        
        // Asumsi kita mengharapkan hasilnya adalah 5.
        // Test ini "flaky" karena kadang berhasil (sekitar 10% probabilitas)
        // dan lebih sering gagal, tanpa ada perubahan kode sama sekali.
        assertEquals(5, result, "Nilai acak yang dihasilkan harus 5");
    }

    /**
     * Contoh Flaky Test 2: Bergantung pada Waktu (Time-bound)
     */
    @Test
    void testRunningOnEvenSecond() {
        long currentSecond = System.currentTimeMillis() / 1000;
        
        // Test ini "flaky" karena hanya akan lulus jika dijalankan tepat pada
        // detik genap, dan akan gagal pada detik ganjil.
        boolean isEven = (currentSecond % 2 == 0);
        
        assertTrue(isEven, "Test harus dijalankan pada detik genap, detik saat ini ganjil!");
    }
}
