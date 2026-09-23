package com.rpla;

import org.junit.jupiter.api.RepeatedTest;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlakyExampleTest {
    private int counter = 0;

    // Ini akan terus berjalan hingga mencapai 100 kali atau sampai ditemukan
    // kegagalan.
    @RepeatedTest(100)
    void testAsyncOperationTimeout() throws InterruptedException {
        // Reset counter untuk setiap perulangan
        counter = 0;
        CountDownLatch completed = new CountDownLatch(1);

        // Mensimulasikan proses asinkron (misal: background job, request API, webhook)
        Thread backgroundTask = new Thread(() -> {
            try {
                // Waktu proses bervariasi, mensimulasikan latensi jaringan atau beban server.
                // Kita ubah rentang waktunya hingga 150 milidetik.
                Thread.sleep((long) (Math.random() * 105));
                counter = 1;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                completed.countDown();
            }
        });

        backgroundTask.start();

        // Menunggu sampai task selesai.
        boolean finished = completed.await(100, TimeUnit.MILLISECONDS);

        assertTrue(finished, "Background task tidak selesai dalam batas waktu (Flaky Test Terjadi!)");
        assertEquals(1, counter, "Counter seharusnya sudah diupdate menjadi 1");
    }
}
