import UIKit
import CoreML
import Vision

class ViewController: UIViewController {
    @IBOutlet weak var ivSelectedImage: UIImageView!
    private let mobilenet = MobileNetV2()
    private lazy var solicitudClasificacion: VNCoreMLRequest = {
        do {
            let model = try VNCoreMLModel(for: mobilenet.model)
            let request = VNCoreMLRequest(
            model: model) { [weak self] request, error in
                guard let self = self else {
                    return
                }
                self.procesarClasificacion(for: request, error: error)
            }
            request.imageCropAndScaleOption = .centerCrop
            return request
        } catch {
            fatalError("Failed to load Vision ML model: \(error)")
        }
    }()
    override func viewDidLoad() {
        super.viewDidLoad()
        navigationController?.navigationBar.prefersLargeTitles = true
        self.navigationController?.navigationBar.largeTitleTextAttributes = [NSAttributedString.Key.foregroundColor: UIColor.white]
    }
    
    @IBAction func loadImage(_ sender: Any) {
        let imagepicker = UIImagePickerController()
        imagepicker.allowsEditing = false
        imagepicker.delegate = self
        present(imagepicker, animated: true, completion: nil)
    }
}
extension ViewController: UINavigationControllerDelegate, UIImagePickerControllerDelegate {
    func imagePickerController(_ picker: UIImagePickerController,
                               didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
        guard let image = info[UIImagePickerController.InfoKey.originalImage] as? UIImage else {
            return
        }
        ivSelectedImage.image = image
        self.iniciarProceso(image)
        picker.dismiss(animated: true, completion: nil)
    }
}
extension ViewController {
    func iniciarProceso(_ image: UIImage) {
        guard let orientation = CGImagePropertyOrientation(
            rawValue: UInt32(image.imageOrientation.rawValue)) else {
                return
        }
        guard let ciImage = CIImage(image: image) else {
            fatalError("Unable to create \(CIImage.self) from \(image).")
        }
        
        DispatchQueue.global(qos: .userInitiated).async {
            let handler = VNImageRequestHandler(ciImage: ciImage, orientation: orientation)
            do {
                try handler.perform([self.solicitudClasificacion])
            } catch {
                print("Failed to perform classification.\n\(error.localizedDescription)")
            }
        }
    }
    func procesarClasificacion(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async { [weak self] in
            guard let results = request.results else {
                self?.title = "Ha ocurrido un error"
                return
            }
            if let classifications = results as? [VNClassificationObservation] {
                let topClassifications = classifications.prefix(2).map {
                    (confidence: $0.confidence, identifier: $0.identifier)
                }
                print("Top classifications: \(topClassifications)")
                let topIdentifiers = topClassifications.map { $0.identifier.lowercased() }
                if let message = topIdentifiers.first {
                    self?.title = message
                }
            }
        }
    }
}
