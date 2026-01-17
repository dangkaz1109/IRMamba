import os
import json

class InfraredSolver(object):
    CLSNAMES = [
        "capsule"
    ]

    def __init__(self, root='data/Infrared'):
        self.root = root
        self.meta_path = f'{root}/meta.json'
        self.phases = ['train', 'test'] 
        self.CLSNAMES = self.CLSNAMES

    def run(self):
        info = {phase: {} for phase in self.phases}
        for cls_name in self.CLSNAMES:
            cls_dir = f'{self.root}/{cls_name}'
            for phase in self.phases:
                cls_info = []
                
               
                phase_dir = f'{cls_dir}/{phase}'
                if not os.path.exists(phase_dir):
                    continue

                species = os.listdir(phase_dir)
                for specie in species:
                
                    specie_full_path = f'{phase_dir}/{specie}'
                   
                    if not os.path.isdir(specie_full_path):
                        continue
                  

                    is_abnormal = True if specie not in ['good'] else False
              
                    img_dir = f'{specie_full_path}/'
                    mask_dir = f'{cls_dir}/ground_truth/{specie}'
                    
                    img_names = os.listdir(img_dir)
                   
                    if is_abnormal and os.path.isdir(mask_dir):
                        mask_names = os.listdir(mask_dir)
                        mask_names.sort()
                    else:
                        mask_names = None

                    img_names.sort()
                    
                    for idx, img_name in enumerate(img_names):
                        
                        current_mask_name = ''
                        if is_abnormal and mask_names and idx < len(mask_names):
                            current_mask_name = mask_names[idx]

                        info_img = dict(
                            img_path=f'{img_dir.replace(cls_dir, cls_name)}{img_name}',
                            mask_path=f'{mask_dir.replace(cls_dir, cls_name)}/{current_mask_name}' if is_abnormal else '',
                            cls_name=cls_name,
                            specie_name=specie,
                            anomaly=1 if is_abnormal else 0,
                        )
                        cls_info.append(info_img)
                info[phase][cls_name] = cls_info
        
        with open(self.meta_path, 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

if __name__ == '__main__':
    runner = InfraredSolver(root='data/Infrared')
    runner.run()
