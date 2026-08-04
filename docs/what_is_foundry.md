# Microsoft Foundry Local Nedir?

Microsoft Foundry Local, yapay zeka (AI) modellerini doğrudan kendi bilgisayarınızda veya cihazınızda (on-device) çalıştırmanıza olanak tanıyan hafif bir çalışma zamanı (runtime) ve yazılım geliştirme kiti (SDK) çözümüdür. 

En büyük avantajı, bulut bağlantısına, bir API anahtarına veya karmaşık sunucu kurulumlarına ihtiyaç duymamasıdır. İlk model indirme işlemi tamamlandıktan sonra, Foundry Local tamamen çevrimdışı (offline) çalışabilir. Bu özellik, veri gizliliğinin önemli olduğu kurumsal uygulamalar veya internet bağlantısının olmadığı saha operasyonları için kritik bir öneme sahiptir.

## Donanım ve Performans
Foundry Local'in bir diğer önemli özelliği ise donanım dostu olmasıdır. Gelişmiş bir ekran kartına (GPU) sahip olmanız şart değildir. Foundry Local, arka planda ONNX Runtime kullanarak bilgisayarınızın işlemcisini (CPU) veya yapay zeka işlemlerine özel sinir ağı işlemcisini (NPU) otomatik olarak tespit eder ve en uygun hızlandırmayı sağlar. Böylece standart dizüstü bilgisayarlarda bile büyük dil modelleri akıcı bir şekilde çalışabilir.

## Desteklenen Diller ve Kurulum
Foundry Local SDK; Python, C#, JavaScript (Node.js) ve Rust gibi popüler programlama dillerini destekler. Python projelerinde, Windows kullanıcıları için "foundry-local-sdk-winml" paketi önerilirken, diğer platformlar (macOS, Linux) için "foundry-local-sdk" paketi kullanılır. Kurulum işlemi standart paket yöneticileri (pip, npm vb.) kullanılarak tek bir komutla kolayca yapılabilir.
