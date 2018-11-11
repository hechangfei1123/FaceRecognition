import net
import train

if __name__ == '__main__':
    ne = net.ONet()
    ne.train()
    trainer = train.Trainer(ne, 'E:/save_path/48/param/',r"E:\save_path\48")
    trainer.trainOnet()