import http.server


class CustomHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.endswith('.mp4'):
            with open(self.path, 'rb') as f:
                self.send_response(200)
                self.send_header('Content-type', 'video/mp4')
                self.end_headers()
                self.wfile.write(f.read())
        else:
            # Serve other files as usual
            super().do_GET()

if __name__ == '__main__':
    server_address = ('', 12345)
    httpd = http.server.HTTPServer(server_address, CustomHTTPHandler)
    httpd.serve_forever()
